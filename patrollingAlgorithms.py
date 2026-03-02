import sys
import os
import numpy as np
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq
import time
import json
from scipy import interpolate
import scipy.ndimage as ndi
from deap import base, creator, tools, algorithms
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
ROBOT_L = 0.5               
ROBOT_RADIUS = ROBOT_L / 2.0 
SAFETY_MARGIN = 0.05        
MIN_SAFE_DIST = ROBOT_RADIUS + SAFETY_MARGIN 

# Non-Holonomic Constraints
# Radius = 2.5m -> Curvature = 0.4
MIN_TURN_RADIUS = 2.5 
MAX_CURVATURE = 1.0 / MIN_TURN_RADIUS
START_TANGENT_LEN = 5.0 

GRID_RES = 0.1
GRID_SIZE = 50

# Optimization resolution 
CHECK_SAMPLES = 500   
PLOT_SAMPLES = 1000

# Global Grid placeholders
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
dist_field = np.zeros((GRID_SIZE, GRID_SIZE))
FIXED_START_DIR = np.array([0.0, 1.0])

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def check_safety_vectorized(path):
    """Objective 1: Safety/Risk (Minimization) - Standard Additive used by GA/NSGA."""
    min_x, min_y = np.min(path[:, 1]), np.min(path[:, 0])
    max_x, max_y = np.max(path[:, 1]), np.max(path[:, 0])
    
    if min_x < 0.5 or min_y < 0.5 or max_x > GRID_SIZE-0.5 or max_y > GRID_SIZE-0.5:
        return 10000 + (abs(min_x)+abs(min_y))*1000

    iy = np.round(path[:, 0]).astype(int)
    ix = np.round(path[:, 1]).astype(int)
    iy = np.clip(iy, 0, GRID_SIZE - 1)
    ix = np.clip(ix, 0, GRID_SIZE - 1)
    
    dists = dist_field[iy, ix] * GRID_RES
    unsafe_mask = dists < MIN_SAFE_DIST
    num_collisions = np.sum(unsafe_mask)
    
    if num_collisions > 0:
        return 10000 + (num_collisions * 100)
    
    safe_dists = dists[~unsafe_mask]
    if len(safe_dists) > 0:
        return np.sum(1.0 / (safe_dists + 0.1)) * 0.01 
    return 0.0

def calculate_violation_paper(path):
    """
    Eq 18 & 19 (Modified): Calculates collision violation v_L for HWPSO.
    FIX: Uses SUM instead of MEAN to ensure thin obstacles are penalized heavily.
    """
    iy = np.round(path[:, 0]).astype(int)
    ix = np.round(path[:, 1]).astype(int)
    iy = np.clip(iy, 0, GRID_SIZE - 1)
    ix = np.clip(ix, 0, GRID_SIZE - 1)
    
    dists = dist_field[iy, ix] * GRID_RES
    
    violations = np.zeros_like(dists)
    unsafe_mask = dists < MIN_SAFE_DIST
    
    if np.any(unsafe_mask):
        # Penalty proportional to depth
        violations[unsafe_mask] = 1.0 - (dists[unsafe_mask] / MIN_SAFE_DIST)
        
    min_x, min_y = np.min(path[:, 1]), np.min(path[:, 0])
    max_x, max_y = np.max(path[:, 1]), np.max(path[:, 0])
    bounds_violation = 0.0
    if min_x < 0 or min_y < 0 or max_x > GRID_SIZE or max_y > GRID_SIZE:
        bounds_violation = 100.0 # Huge penalty for leaving map
        
    # FIX: Use SUM instead of MEAN. 
    # This represents the "Total Length" traveled inside an obstacle.
    v_L = np.sum(violations) + bounds_violation
    return v_L

def check_forward_motion(path, start_dir):
    if len(path) < 2: return 0.0
    idx = min(len(path)-1, max(1, int(len(path)*0.05)))
    vec = path[idx] - path[0] 
    norm = np.linalg.norm(vec)
    if norm < 1e-3: return 0.0
    dot = np.dot(vec/norm, start_dir)
    if dot < 0.0: return 1000.0 * (abs(dot) + 1.0)
    return 0.0

def get_path_length(path):
    if len(path) < 2: return 0.0
    return np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))) * GRID_RES

def _curvature_array_physical(path):
    """Compute curvature array in physical units (1/m) using arc-length resampling.
    Returns (curvatures, spacing) for a uniformly-spaced resample of the path."""
    if len(path) < 3:
        return np.array([0.0]), 1.0
    # Convert to physical coordinates (meters)
    phys = path * GRID_RES
    # Compute arc length
    diffs = np.diff(phys, axis=0)
    seg_lens = np.sqrt(np.sum(diffs**2, axis=1))
    cum_len = np.concatenate(([0], np.cumsum(seg_lens)))
    total_len = cum_len[-1]
    if total_len < 0.05:
        return np.array([0.0]), 1.0
    # Resample at uniform arc-length spacing (~0.2m)
    spacing = 0.2
    n_pts = max(10, int(total_len / spacing))
    s_new = np.linspace(0, total_len, n_pts)
    x_new = np.interp(s_new, cum_len, phys[:, 1])
    y_new = np.interp(s_new, cum_len, phys[:, 0])
    # Compute curvature using np.gradient with correct spacing
    ds = s_new[1] - s_new[0]  # uniform
    dy = np.gradient(y_new, ds)
    dx = np.gradient(x_new, ds)
    ddy = np.gradient(dy, ds)
    ddx = np.gradient(dx, ds)
    num = np.abs(dx * ddy - dy * ddx)
    den = np.power(dx**2 + dy**2, 1.5)
    den[den < 1e-12] = 1e-12
    curvatures = num / den
    # Exclude 2 points at each end (gradient boundary effects)
    if len(curvatures) > 6:
        curvatures = curvatures[3:-3]
    return curvatures, ds

def get_max_curvature(path):
    """Compute max curvature in physical units (1/m) with robust arc-length resampling."""
    curvatures, _ = _curvature_array_physical(path)
    return float(np.max(curvatures))

def get_curvature_penalty(path):
    """
    Gradient-based penalty for Non-Holonomic Constraints.
    Curvature computed in physical units (1/m) vs MAX_CURVATURE (1/m).
    """
    max_k = get_max_curvature(path)
    if max_k > MAX_CURVATURE:
        return (max_k - MAX_CURVATURE) * 5000.0
    return 0.0

def get_curvature_integral_penalty(path):
    """Sum over all points where curvature exceeds MAX_CURVATURE.
    Curvature computed in physical units (1/m) with arc-length resampling."""
    curvatures, ds = _curvature_array_physical(path)
    excess = curvatures - MAX_CURVATURE
    excess[excess < 0] = 0
    # Weight by arc-length segment (ds) so penalty scales with path length
    return float(np.sum(excess) * ds * 500.0)

def get_smoothness_cost(path):
    if len(path) < 3: return 0.0
    vecs = np.diff(path, axis=0)
    norms = np.linalg.norm(vecs, axis=1)
    norms[norms < 1e-6] = 1.0
    vecs = vecs / norms[:, None]
    dots = np.sum(vecs[:-1] * vecs[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    return np.sum(angles) * 10.0

def common_spline(control_points, num_samples=100, smoothing=0):
    pts = np.array(control_points)
    if len(pts) < 2: return pts
    if len(pts) == 2: return np.linspace(pts[0], pts[1], num_samples)
    keep = [pts[0]]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - keep[-1]) > 0.1: keep.append(pts[i])
    pts = np.array(keep)
    if len(pts) < 3: return np.linspace(pts[0], pts[-1], num_samples)
    k = 3 
    y = pts[:, 0]; x = pts[:, 1]
    try:
        tck, u = interpolate.splprep([x, y], s=smoothing, k=k)
        u_new = np.linspace(0, 1, num_samples)
        x_new, y_new = interpolate.splev(u_new, tck, der=0)
        result = np.column_stack((y_new, x_new))
        # Clamp to grid boundaries to prevent spline overshoot
        result[:, 0] = np.clip(result[:, 0], 0, GRID_SIZE - 1)
        result[:, 1] = np.clip(result[:, 1], 0, GRID_SIZE - 1)
        # Pin endpoints: smoothing>0 creates approximating spline that drifts from endpoints
        result[0] = pts[0]
        result[-1] = pts[-1]
        return result
    except:
        return np.linspace(pts[0], pts[-1], num_samples)

def sparsify_path(path, min_dist=3.0):
    if len(path) < 2: return path
    new_path = [path[0]]
    for p in path[1:-1]:
        if np.linalg.norm(np.array(p) - np.array(new_path[-1])) > min_dist:
            new_path.append(p)
    new_path.append(path[-1])
    return np.array(new_path)

def curvature_limit_respline(path, num_samples=None):
    """Re-spline the path with increasing smoothing until curvature ≤ MAX_CURVATURE.
    Returns the smoothest collision-free version found."""
    if num_samples is None:
        num_samples = len(path) if len(path) >= PLOT_SAMPLES else PLOT_SAMPLES
    safe_threshold = MIN_SAFE_DIST / GRID_RES
    best = path
    best_k = get_max_curvature(path)
    if best_k <= MAX_CURVATURE:
        return best
    # Try increasing smoothing factors
    sparse = sparsify_path(path, min_dist=3.0)
    if len(sparse) < 3:
        return best
    for s_factor in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        s_val = len(sparse) * s_factor
        candidate = common_spline(sparse, num_samples, smoothing=s_val)
        # Verify collision safety
        ciy = np.clip(np.round(candidate[:, 0]).astype(int), 0, GRID_SIZE - 1)
        cix = np.clip(np.round(candidate[:, 1]).astype(int), 0, GRID_SIZE - 1)
        if np.any(grid[ciy, cix] == 1):
            continue  # spline went through obstacle
        if np.any(dist_field[ciy, cix] < safe_threshold * 0.7):
            continue  # too close to obstacle
        k = get_max_curvature(candidate)
        if k < best_k:
            best = candidate
            best_k = k
        if best_k <= MAX_CURVATURE:
            break
    return best

def smooth_path_kinematic(path, iterations=200, alpha=0.15, beta=0.50):
    """Vectorized iterative smoother that respects collision constraints.
    Moves interior points toward the average of their neighbours while
    checking that new positions stay collision-free."""
    if len(path) < 5:
        return path
    safe_threshold = MIN_SAFE_DIST / GRID_RES
    smoothed = path.copy().astype(float)
    n = len(smoothed)
    for _ in range(iterations):
        # Vectorized 5-point stencil on interior [2 .. n-3]
        avg_far = (smoothed[:-4] + smoothed[1:-3] + smoothed[3:-1] + smoothed[4:]) / 4.0
        avg_near = (smoothed[1:-3] + smoothed[3:-1]) / 2.0
        interior = smoothed[2:-2]
        candidates = interior + alpha * (avg_far - interior) + beta * (avg_near - interior)
        # Clamp to grid
        candidates[:, 0] = np.clip(candidates[:, 0], 0, GRID_SIZE - 1)
        candidates[:, 1] = np.clip(candidates[:, 1], 0, GRID_SIZE - 1)
        # Check collision safety for all candidates at once
        cy = np.clip(np.round(candidates[:, 0]).astype(int), 0, GRID_SIZE - 1)
        cx = np.clip(np.round(candidates[:, 1]).astype(int), 0, GRID_SIZE - 1)
        safe_mask = dist_field[cy, cx] >= safe_threshold
        # Only update safe candidates
        new_interior = interior.copy()
        new_interior[safe_mask] = candidates[safe_mask]
        smoothed[2:-2] = new_interior
    return smoothed


def enforce_max_curvature(path, max_iters=20):
    """Directly enforce MAX_CURVATURE on the path by iteratively adjusting
    points where the turning angle exceeds the kinematic limit.
    Uses the discrete approximation: curvature ≈ turning_angle / segment_length.
    Collision-aware: only accepts moves that stay safe.
    Operates on ALL interior points including near-endpoints (indices 1..n-2)."""
    if len(path) < 3:
        return path
    safe_threshold = MIN_SAFE_DIST / GRID_RES
    result = path.copy().astype(float)
    n = len(result)

    for iteration in range(max_iters):
        # Compute segment vectors and lengths
        segs = np.diff(result, axis=0)  # (n-1, 2)
        seg_lens = np.linalg.norm(segs, axis=1) * GRID_RES  # physical metres
        seg_lens = np.maximum(seg_lens, 1e-8)

        # Unit direction vectors
        dirs = segs / np.linalg.norm(segs, axis=1, keepdims=True).clip(1e-8)

        # Turning angles at interior points
        dots = np.sum(dirs[:-1] * dirs[1:], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # (n-2,) at points 1..n-2

        # Average segment length at each interior point
        avg_seg = (seg_lens[:-1] + seg_lens[1:]) / 2.0  # (n-2,)

        # Curvature approximation at interior points
        curvatures = angles / avg_seg  # (n-2,)

        # Find violations (excluding first and last few points to avoid
        # messing with pinned endpoints)
        violations = np.where(curvatures > MAX_CURVATURE)[0]  # indices into [1..n-2]

        if len(violations) == 0:
            break  # All curvatures within limit

        # Process violations: move each violating point toward neighbour midpoint
        for vi in violations:
            i = vi + 1  # actual index in result array
            if i < 1 or i >= n - 1:
                continue  # don't modify endpoints themselves

            # Target: midpoint of neighbours
            mid = (result[i - 1] + result[i + 1]) / 2.0
            # Move fraction toward midpoint (stronger with higher curvature)
            excess_ratio = min(curvatures[vi] / MAX_CURVATURE, 5.0)
            move_frac = min(0.5, 0.15 * excess_ratio)
            candidate = result[i] + move_frac * (mid - result[i])

            candidate[0] = np.clip(candidate[0], 0, GRID_SIZE - 1)
            candidate[1] = np.clip(candidate[1], 0, GRID_SIZE - 1)
            cy = int(np.clip(round(candidate[0]), 0, GRID_SIZE - 1))
            cx = int(np.clip(round(candidate[1]), 0, GRID_SIZE - 1))
            if dist_field[cy, cx] >= safe_threshold:
                result[i] = candidate

    return result


def _escape_obstacles(path):
    """Move any path points that sit on actual obstacle cells to the nearest free cell."""
    result = path.copy()
    for i in range(len(result)):
        gy = int(np.clip(round(result[i, 0]), 0, GRID_SIZE - 1))
        gx = int(np.clip(round(result[i, 1]), 0, GRID_SIZE - 1))
        if grid[gy, gx] == 1:
            # Search expanding ring for nearest free cell
            found = False
            for r in range(1, 8):
                best_dist = 1e9
                best_pt = None
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ny, nx = gy + dy, gx + dx
                        if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                            if grid[ny, nx] == 0 and dist_field[ny, nx] >= 1.5:
                                d = (dy**2 + dx**2)**0.5
                                if d < best_dist:
                                    best_dist = d
                                    best_pt = (float(ny), float(nx))
                if best_pt is not None:
                    result[i] = np.array(best_pt)
                    found = True
                    break
            if not found:
                # Extreme fallback: use distance field gradient direction
                pass
    return result


def ensure_collision_free(path):
    """Post-process: replace collision segments with A* detours,
    smooth A* detours via spline before stitching, and apply final
    kinematic smoothing to the whole path."""
    if len(path) < 2:
        return path

    # Full safety threshold for detection
    safe_threshold = MIN_SAFE_DIST / GRID_RES  # 3.0 cells

    iy = np.clip(np.round(path[:, 0]).astype(int), 0, GRID_SIZE - 1)
    ix = np.clip(np.round(path[:, 1]).astype(int), 0, GRID_SIZE - 1)
    in_collision = dist_field[iy, ix] < safe_threshold

    if not np.any(in_collision):
        # Already collision-free — apply curvature-limiting re-spline + kinematic smoothing
        result = curvature_limit_respline(path)
        result = smooth_path_kinematic(result)
        result = enforce_max_curvature(result, max_iters=20)
        result = _escape_obstacles(result)
        # Pin endpoints to match original path start/end
        result[0] = path[0]
        result[-1] = path[-1]
        return result

    # Detect collision segment boundaries
    padded = np.concatenate(([False], in_collision, [False]))
    diffs = np.diff(padded.astype(int))
    seg_starts = np.where(diffs == 1)[0]
    seg_ends   = np.where(diffs == -1)[0]

    parts = []
    prev = 0

    # Inflation levels to try (from safest to minimal)
    full_inflate = int(np.ceil(safe_threshold))  # 3
    inflate_levels = [full_inflate, 2, 1]

    for s, e in zip(seg_starts, seg_ends):
        # Keep safe segment before collision
        if prev < s:
            parts.append(path[prev:s])

        # Entry = last safe point before collision, Exit = first safe point after
        entry_idx = max(0, s - 1)
        exit_idx  = min(len(path) - 1, e)

        # Walk outward to ensure entry/exit are truly safe
        while entry_idx > 0 and in_collision[entry_idx]:
            entry_idx -= 1
        while exit_idx < len(path) - 1 and in_collision[exit_idx]:
            exit_idx += 1

        sg = (int(np.clip(round(path[entry_idx, 0]), 0, GRID_SIZE - 1)),
              int(np.clip(round(path[entry_idx, 1]), 0, GRID_SIZE - 1)))
        eg = (int(np.clip(round(path[exit_idx, 0]), 0, GRID_SIZE - 1)),
              int(np.clip(round(path[exit_idx, 1]), 0, GRID_SIZE - 1)))

        # Cascading A* with decreasing inflation
        detour = None
        for inflate in inflate_levels:
            detour = a_star(grid, sg, eg, inflation=inflate)
            if detour is not None:
                break

        if detour and len(detour) >= 2:
            # Smooth the A* detour through sparsify + spline so it's not jagged
            detour_arr = np.array(detour, dtype=float)
            if len(detour_arr) >= 4:
                sparse = sparsify_path(detour_arr, min_dist=2.0)
                if len(sparse) >= 3:
                    splined = common_spline(sparse, num_samples=max(50, len(detour_arr)*2))
                    # Verify the splined detour is collision-free
                    siy = np.clip(np.round(splined[:, 0]).astype(int), 0, GRID_SIZE - 1)
                    six = np.clip(np.round(splined[:, 1]).astype(int), 0, GRID_SIZE - 1)
                    if np.all(dist_field[siy, six] >= safe_threshold):
                        detour_arr = splined
            parts.append(detour_arr)
        else:
            # Last resort: keep original (should be very rare)
            parts.append(path[entry_idx:exit_idx + 1])

        prev = exit_idx

    # Add remaining safe part after last collision
    if prev < len(path):
        parts.append(path[prev:])

    if not parts:
        return path

    stitched = np.vstack(parts)

    # Remove near-duplicate consecutive points
    keep = [0]
    for i in range(1, len(stitched)):
        if np.linalg.norm(stitched[i] - stitched[keep[-1]]) > 0.05:
            keep.append(i)
    if keep[-1] != len(stitched) - 1:
        keep.append(len(stitched) - 1)
    stitched = stitched[np.array(keep)]

    if len(stitched) < 2:
        return path

    # Re-spline through sparse waypoints for smooth transitions
    sparse_stitch = sparsify_path(stitched, min_dist=2.0)
    if len(sparse_stitch) >= 3:
        resplined = common_spline(sparse_stitch, PLOT_SAMPLES)
        # Verify collision safety of resplined path
        riy = np.clip(np.round(resplined[:, 0]).astype(int), 0, GRID_SIZE - 1)
        rix = np.clip(np.round(resplined[:, 1]).astype(int), 0, GRID_SIZE - 1)
        if np.all(dist_field[riy, rix] >= safe_threshold * 0.9):
            resampled = resplined
        else:
            # Fallback: linear resample
            seg_lens = np.sqrt(np.sum(np.diff(stitched, axis=0)**2, axis=1))
            cum_len = np.concatenate(([0], np.cumsum(seg_lens)))
            total = cum_len[-1]
            if total < 1e-6: return path
            t_new = np.linspace(0, total, PLOT_SAMPLES)
            new_y = np.interp(t_new, cum_len, stitched[:, 0])
            new_x = np.interp(t_new, cum_len, stitched[:, 1])
            resampled = np.column_stack((new_y, new_x))
    else:
        # Too few points for spline, use linear resample
        seg_lens = np.sqrt(np.sum(np.diff(stitched, axis=0)**2, axis=1))
        cum_len = np.concatenate(([0], np.cumsum(seg_lens)))
        total = cum_len[-1]
        if total < 1e-6: return path
        t_new = np.linspace(0, total, PLOT_SAMPLES)
        new_y = np.interp(t_new, cum_len, stitched[:, 0])
        new_x = np.interp(t_new, cum_len, stitched[:, 1])
        resampled = np.column_stack((new_y, new_x))

    # Final kinematic smoothing + curvature limiting
    resampled = _escape_obstacles(resampled)
    resampled = curvature_limit_respline(resampled)
    smoothed = smooth_path_kinematic(resampled)
    smoothed = enforce_max_curvature(smoothed, max_iters=20)
    smoothed = _escape_obstacles(smoothed)
    # Pin endpoints to match original path start/end
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    return smoothed

# --- NSGA SPECIFIC OBJECTIVES ---
def calculate_wheel_effort(path_coords):
    if len(path_coords) < 2: return float('inf')
    total_energy = 0.0; R_WHEEL = 0.1 
    for i in range(len(path_coords) - 1):
        p1, p2 = path_coords[i], path_coords[i+1]
        ds = np.linalg.norm(p1 - p2) * GRID_RES
        d_theta = 0
        if i < len(path_coords) - 2:
            p3 = path_coords[i+2]
            v1 = p2-p1; v2 = p3-p2
            a1 = np.arctan2(v1[0], v1[1]); a2 = np.arctan2(v2[0], v2[1])
            d_theta = (a2 - a1 + np.pi) % (2*np.pi) - np.pi
        ds_R = ds + (ROBOT_L / 2.0) * d_theta
        ds_L = ds - (ROBOT_L / 2.0) * d_theta
        total_energy += ( (ds_R/R_WHEEL)**2 + (ds_L/R_WHEEL)**2 )
    return total_energy

def calculate_centering_score(path_coords, obstacles_list):
    if len(obstacles_list) == 0: return 0.0
    obs_array = np.array(obstacles_list)
    total_risk = 0.0
    p_start, p_end = path_coords[0], path_coords[-1]
    RELAX = 1.5 
    for p in path_coords[::5]: 
        d_s = np.linalg.norm(p - p_start) * GRID_RES
        d_e = np.linalg.norm(p - p_end) * GRID_RES
        w = 0.0 if (d_s < RELAX or d_e < RELAX) else 1.0
        if w > 0:
            dists = np.sqrt(np.sum((obs_array - p)**2, axis=1)) * GRID_RES
            min_d = np.min(dists)
            if min_d < 0.05: min_d = 0.05
            total_risk += (1.0 / min_d) * w
    return total_risk

def construct_adaptive_path(champion_inds, obstacles_list):
    """Build adaptive path as a smooth weighted blend of champion paths.

    Instead of discrete point-by-point selection (which causes abrupt
    switches and sharp turns), this computes per-point quality scores for
    each champion, applies heavy temporal smoothing so the weights change
    gradually, and produces a soft-weighted blend of all champion paths.
    This inherently produces kinematically-feasible paths for non-holonomic
    two-wheeled robots because the output varies smoothly in space."""
    interpolated_paths = []
    for ind in champion_inds:
        pts = common_spline(ind, num_samples=PLOT_SAMPLES)
        interpolated_paths.append(pts)

    min_len = min(len(p) for p in interpolated_paths)
    # Shape: (n_champs, min_len, 2)
    paths_arr = np.array([p[:min_len] for p in interpolated_paths])
    n_champs = paths_arr.shape[0]

    # ---- Vectorized per-point quality scores (lower = better) ----
    scores = np.zeros((n_champs, min_len))
    for ci in range(n_champs):
        iy = np.clip(np.round(paths_arr[ci, :, 0]).astype(int), 0, GRID_SIZE - 1)
        ix = np.clip(np.round(paths_arr[ci, :, 1]).astype(int), 0, GRID_SIZE - 1)
        oob = ((paths_arr[ci, :, 0] < 0) | (paths_arr[ci, :, 0] >= GRID_SIZE) |
               (paths_arr[ci, :, 1] < 0) | (paths_arr[ci, :, 1] >= GRID_SIZE))
        dists = dist_field[iy, ix] * GRID_RES
        unsafe = dists < MIN_SAFE_DIST
        scores[ci] = 0.5 / (dists + 0.1) + 0.5 / (dists + 0.01)
        scores[ci, unsafe] = 50.0
        scores[ci, oob] = 100.0

    # Heavy temporal smoothing — prevents rapid weight fluctuation
    kernel = max(min_len // 5, 80)
    for ci in range(n_champs):
        scores[ci] = ndi.uniform_filter1d(scores[ci].astype(float),
                                          size=kernel, mode='nearest')

    # Softmin weights (low temperature → sharper selection of best champion)
    temperature = 0.3
    neg = -scores / temperature
    neg -= np.max(neg, axis=0, keepdims=True)  # numerical stability
    w = np.exp(neg)
    weights = w / np.maximum(np.sum(w, axis=0, keepdims=True), 1e-12)

    # Weighted blend of all champion paths
    adaptive = np.einsum('ct,ctd->td', weights, paths_arr)

    # Subsample to ~100 control points for downstream spline fitting
    step = max(min_len // 100, 1)
    result = adaptive[::step]
    if len(result) > 2:
        result[0] = adaptive[0]
        result[-1] = adaptive[-1]

    # Collision-aware iterative smoothing on control points
    if len(result) >= 5:
        for _ in range(60):
            new_r = result.copy()
            for i in range(2, len(result) - 2):
                avg = (result[i-2] + result[i-1] +
                       result[i+1] + result[i+2]) / 4.0
                cand = result[i] + 0.40 * (avg - result[i])
                cand[0] = np.clip(cand[0], 0, GRID_SIZE - 1)
                cand[1] = np.clip(cand[1], 0, GRID_SIZE - 1)
                cy = int(np.clip(round(cand[0]), 0, GRID_SIZE - 1))
                cx = int(np.clip(round(cand[1]), 0, GRID_SIZE - 1))
                if dist_field[cy, cx] * GRID_RES >= MIN_SAFE_DIST:
                    new_r[i] = cand
            result = new_r
        result[0] = adaptive[0]
        result[-1] = adaptive[-1]

    return result

# ---------------------------------------------------------
# 3. HELPER ALGORITHMS (A* STAR)
# ---------------------------------------------------------
def a_star(current_grid, start, goal, inflation=None):
    if inflation is not None:
        safe_cells = inflation
    else:
        safe_cells = int(np.ceil(MIN_SAFE_DIST / GRID_RES))
    inflated_grid = np.zeros_like(current_grid)
    inflated_grid[dist_field < safe_cells] = 1
    # Ensure start and goal are accessible
    if inflated_grid[start[0], start[1]] == 1 or inflated_grid[goal[0], goal[1]] == 1:
        return None
    def h(a, b): return np.hypot(a[0]-b[0], a[1]-b[1])
    open_set = []; heapq.heappush(open_set, (0, start))
    came_from = {}; g_score = {start: 0}
    while open_set:
        curr = heapq.heappop(open_set)[1]
        if curr == goal:
            path = []
            while curr in came_from: path.append(curr); curr = came_from[curr]
            path.append(start)
            return path[::-1]
        for dy, dx in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            ny, nx = curr[0]+dy, curr[1]+dx
            if 0<=ny<GRID_SIZE and 0<=nx<GRID_SIZE and inflated_grid[ny,nx]==0:
                tg = g_score[curr] + np.hypot(dy, dx)
                if tg < g_score.get((ny,nx), float('inf')):
                    came_from[(ny,nx)] = curr
                    g_score[(ny,nx)] = tg
                    heapq.heappush(open_set, (tg + h((ny,nx), goal), (ny,nx)))
    return None

def _safe_tangent_point(anchor, direction, max_len=None):
    """Compute an obstacle-aware tangent control point for spline direction.
    
    Places the point at  anchor + max_len * direction.  If that position
    falls inside or too close to an obstacle, progressively shrinks the
    distance until a collision-free location is found.
    
    This is critical for non-holonomic heading continuity: the tangent
    point steers the spline's first-derivative at the anchor, ensuring
    the robot departs/arrives in the intended direction.
    
    Args:
        anchor:    Base waypoint (y, x) in grid coordinates.
        direction: Unit direction vector (dy, dx).
        max_len:   Maximum tangent offset (grid cells). Defaults to START_TANGENT_LEN.
    Returns:
        Tangent point as numpy array, guaranteed collision-free.
    """
    if max_len is None:
        max_len = START_TANGENT_LEN
    safe_threshold = max(1.5, (MIN_SAFE_DIST / GRID_RES) * 0.5)
    anchor = np.array(anchor, dtype=float)
    direction = np.array(direction, dtype=float)
    for frac in [1.0, 0.75, 0.5, 0.35, 0.2]:
        pt = anchor + (max_len * frac) * direction
        pt[0] = np.clip(pt[0], 0.5, GRID_SIZE - 1.5)
        pt[1] = np.clip(pt[1], 0.5, GRID_SIZE - 1.5)
        gy = int(np.clip(round(pt[0]), 0, GRID_SIZE - 1))
        gx = int(np.clip(round(pt[1]), 0, GRID_SIZE - 1))
        if grid[gy, gx] == 0 and dist_field[gy, gx] >= safe_threshold:
            return pt
    # Fallback: return anchor (no tangent effect, but at least safe)
    return anchor.copy()

# ---------------------------------------------------------
# 4. ALGORITHMS
# ---------------------------------------------------------

# --- 1. STANDARD GA ---
def run_ga_standard(start_node, goal_node, start_dir=None, goal_dir=None):
    if hasattr(creator, "FitnessGA"): del creator.FitnessGA
    if hasattr(creator, "IndividualGA"): del creator.IndividualGA
    creator.create("FitnessGA", base.Fitness, weights=(-1.0,))
    creator.create("IndividualGA", list, fitness=creator.FitnessGA)

    start_pts = [np.array(start_node)]
    if start_dir is not None:
        start_pts.append(_safe_tangent_point(start_node, start_dir))

    # Goal tangent: ensure the path approaches the goal heading toward goal_dir
    goal_pts = []
    if goal_dir is not None:
        goal_pts.append(tuple(_safe_tangent_point(goal_node, -np.array(goal_dir))))
    goal_pts.append(goal_node)

    gen_counter = [0]
    MAX_GENS = 40

    def eval_ga(individual):
        sorted_ind = sorted(individual, key=lambda p: np.linalg.norm(np.array(p) - np.array(start_node)))
        full_pts = start_pts + sorted_ind + goal_pts
        path = common_spline(full_pts, CHECK_SAMPLES)
        length = get_path_length(path)
        safety = check_safety_vectorized(path)
        smooth = get_smoothness_cost(path)
        kinematic_penalty = get_curvature_penalty(path) + get_curvature_integral_penalty(path)
        cost = 30.0 * length + 1000.0 * safety + 10.0 * smooth + kinematic_penalty
        if start_dir is not None: cost += check_forward_motion(path, start_dir)
        return (cost,)

    def mut_ga(ind, indpb=0.2):
        current_gen = gen_counter[0]
        sigma = 3.0 * (1.0 - (current_gen / MAX_GENS)) + 0.1
        for i in range(len(ind)):
            if random.random() < indpb:
                ind[i] = (np.clip(ind[i][0] + random.gauss(0, sigma), 0, GRID_SIZE-1),
                          np.clip(ind[i][1] + random.gauss(0, sigma), 0, GRID_SIZE-1))
        return ind,

    toolbox = base.Toolbox()
    def init_ind():
        pts = []
        for i in range(1, 6):
            alpha = i / 6.0
            y = (1-alpha)*start_node[0] + alpha*goal_node[0] + random.uniform(-4,4)
            x = (1-alpha)*start_node[1] + alpha*goal_node[1] + random.uniform(-4,4)
            pts.append((np.clip(y,0,GRID_SIZE-1), np.clip(x,0,GRID_SIZE-1)))
        return creator.IndividualGA(pts)

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_ga)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mut_ga)
    toolbox.register("select", tools.selTournament, tournsize=5)

    hof = tools.HallOfFame(1)
    pop = toolbox.population(n=50)
    for g in range(MAX_GENS):
        gen_counter[0] = g
        pop = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)
        fits = toolbox.map(toolbox.evaluate, pop)
        for fit, ind in zip(fits, pop): ind.fitness.values = fit
        hof.update(pop)
        pop = toolbox.select(pop, len(pop))
    best = sorted(hof[0], key=lambda p: np.linalg.norm(np.array(p) - np.array(start_node)))
    final_path = common_spline(start_pts + best + goal_pts, PLOT_SAMPLES)
    final_path = ensure_collision_free(final_path)
    # Pin endpoints to exact waypoints
    final_path[0] = np.array(start_node, dtype=float)
    final_path[-1] = np.array(goal_node, dtype=float)
    
    if len(final_path) > 50:
        end_dir = final_path[-1] - final_path[-50]
        norm = np.linalg.norm(end_dir)
        end_dir = end_dir / norm if norm > 0 else np.array([0,1])
    else: end_dir = np.array([0,1])
    return final_path, end_dir

# --- 2. BIO-HYBRID COMPETITOR (Paper 1) ---
def run_aco_initialization(grid, start, goal):
    rows, cols = grid.shape
    pheromone = np.ones((rows, cols))
    NUM_ANTS = 30
    ITERATIONS = 20
    EVAPORATION = 0.3
    ALPHA = 1.0
    BETA = 1.5 
    best_path_nodes = None
    best_len = float('inf')
    directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    for it in range(ITERATIONS):
        paths = []
        for ant in range(NUM_ANTS):
            current = start
            path = [current]
            visited = set()
            visited.add(current)
            steps = 0
            while current != goal and steps < rows*cols:
                neighbors = []
                probs = []
                for dy, dx in directions:
                    ny, nx = current[0]+dy, current[1]+dx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        if grid[ny, nx] == 0 and (ny, nx) not in visited:
                            dist = np.sqrt((ny-goal[0])**2 + (nx-goal[1])**2)
                            h = 1.0 / (dist + 1.0) 
                            p = (pheromone[ny, nx] ** ALPHA) * (h ** BETA)
                            neighbors.append((ny, nx))
                            probs.append(p)
                if not neighbors: break 
                probs = np.array(probs) / np.sum(probs)
                next_node = neighbors[np.random.choice(len(neighbors), p=probs)]
                current = next_node
                path.append(current)
                visited.add(current)
                steps += 1
            if current == goal:
                paths.append(path)
                if len(path) < best_len:
                    best_len = len(path); best_path_nodes = path
        pheromone *= (1.0 - EVAPORATION)
        for p in paths:
            score = 50.0 / len(p)
            for (r, c) in p: pheromone[r, c] += score
    if best_path_nodes is None: return [start, goal] 
    return best_path_nodes

def solve_bio_hybrid_competitor(grid, inflated_grid, obstacles_list, start, goal):
    """
    AB-WOA: ACO-Based Whale Optimization Algorithm + APF
    Paper: "Bio-inspired hybrid path planning for efficient and smooth robotic navigation"
    
    Phase 1: ACO initialization for a feasible initial path
    Phase 2: Population-based WOA with APF integration
             - Three WOA mechanisms: encircling (|A|<1), spiral (p>=0.5), exploration (|A|>=1)
             - APF: attractive force toward goal + repulsive force from obstacles
             - Smoothing via neighbor midpoint averaging
             - Random jump mechanism to escape local minima
             - Greedy acceptance: new position accepted only if it improves cost
    Phase 3: B-spline smoothing (handled downstream via common_spline)
    """
    print("  > Path 8: ACO Initialization...")
    path_nodes = run_aco_initialization(inflated_grid, start, goal)
    obs_array = np.array(obstacles_list)

    # --- Population-based WOA + APF Optimization ---
    N_WHALES = 20
    ITERATIONS = 50
    MAX_STEP = 3.0  # Clamp maximum displacement per iteration (grid cells)

    # Sample interior control points from ACO path
    aco_path = np.array(path_nodes, dtype=float)
    if len(aco_path) < 7:
        ts = np.linspace(0, 1, 15)
        aco_path = np.array(start, dtype=float) + np.outer(ts, np.array(goal, dtype=float) - np.array(start, dtype=float))

    N_CTRL = min(8, max(5, len(aco_path) // 5))
    indices = np.linspace(1, len(aco_path) - 2, N_CTRL).astype(int)
    base_ctrl = aco_path[indices]

    # Initialize whale population by perturbing ACO control points
    population = np.zeros((N_WHALES, N_CTRL, 2))
    for i in range(N_WHALES):
        for j in range(N_CTRL):
            noise = np.random.uniform(-3, 3, 2)
            population[i, j] = np.clip(base_ctrl[j] + noise, 0, GRID_SIZE - 1)

    # APF parameters (scaled for grid-cell units)
    K_ATT = 0.3    # Attractive gain (ξ) — moderate pull toward goal
    K_REP = 1.0    # Repulsive gain (η) — obstacle push
    D0 = 4.0       # Repulsive influence distance (grid cells)
    APF_SCALE = 0.15  # Scale factor for APF force application

    start_arr = np.array(start, dtype=float)
    goal_arr = np.array(goal, dtype=float)

    def evaluate_whale(ctrl_pts):
        """Evaluate a whale (set of control points) as a full spline path."""
        sorted_pts = ctrl_pts[np.argsort(np.linalg.norm(ctrl_pts - start_arr, axis=1))]
        full_pts = np.vstack([start_arr, sorted_pts, goal_arr])
        path = common_spline(full_pts, CHECK_SAMPLES)
        length = get_path_length(path)
        safety = check_safety_vectorized(path)
        smooth = get_smoothness_cost(path)
        curv = get_curvature_penalty(path) + get_curvature_integral_penalty(path)
        return 30.0 * length + 1000.0 * safety + 10.0 * smooth + curv

    def compute_apf_force(point):
        """Compute combined APF force: attractive + repulsive.
        Attractive: F_att = ξ · (q_goal - q) / |q_goal - q|
        Repulsive:  F_rep = η · (1/d - 1/d0) · (1/d²) · n̂   when d < d0
        Force magnitude is clamped to avoid explosive jumps near obstacles.
        """
        # Attractive force toward goal
        vec_to_goal = goal_arr - point
        dist_to_goal = np.linalg.norm(vec_to_goal)
        if dist_to_goal > 1e-3:
            f_att = K_ATT * vec_to_goal / dist_to_goal
        else:
            f_att = np.zeros(2)

        # Repulsive force from nearest obstacle
        f_rep = np.zeros(2)
        if len(obs_array) > 0:
            dists = np.sqrt(np.sum((obs_array - point) ** 2, axis=1))
            min_idx = np.argmin(dists)
            min_d = max(dists[min_idx], 0.5)  # Floor to prevent explosion
            if min_d < D0:
                vec_from_obs = point - obs_array[min_idx]
                norm_v = np.linalg.norm(vec_from_obs)
                if norm_v > 1e-6:
                    unit_vec = vec_from_obs / norm_v
                    rep_mag = K_REP * (1.0/min_d - 1.0/D0) * (1.0/(min_d**2))
                    rep_mag = min(rep_mag, 3.0)  # Clamp repulsive force magnitude
                    f_rep = rep_mag * unit_vec

        return f_att + f_rep

    # Find initial global best whale and per-whale personal bests
    gbest = population[0].copy()
    gbest_cost = float('inf')
    whale_costs = np.full(N_WHALES, float('inf'))
    for i in range(N_WHALES):
        whale_costs[i] = evaluate_whale(population[i])
        if whale_costs[i] < gbest_cost:
            gbest_cost = whale_costs[i]
            gbest = population[i].copy()

    print("  > Path 8: WOA-APF Optimization...")
    for t in range(ITERATIONS):
        # WOA coefficient: a linearly decreases from 2 to 0
        a_coeff = 2.0 - 2.0 * t / ITERATIONS
        # Adaptive step size: large early (exploration), small late (exploitation)
        step_limit = MAX_STEP * (1.0 - 0.5 * t / ITERATIONS)

        for i in range(N_WHALES):
            new_ctrl = np.copy(population[i])
            for j in range(N_CTRL):
                r1 = np.random.rand()
                r2 = np.random.rand()
                A_vec = 2 * a_coeff * r1 - a_coeff   # Eq: A = 2a·r - a
                C_vec = 2 * r2                        # Eq: C = 2r
                p = np.random.rand()
                l_param = np.random.uniform(-1, 1)
                b_spiral = 1.0
                curr = population[i, j]

                if p < 0.5:
                    if abs(A_vec) < 1:
                        # Phase 1: Encircling prey (Eq 10)
                        D = abs(C_vec * gbest[j] - curr)
                        woa_pos = gbest[j] - A_vec * D
                    else:
                        # Phase 3: Exploration - random whale (Eq 14)
                        rand_idx = np.random.randint(0, N_WHALES)
                        D = abs(C_vec * population[rand_idx, j] - curr)
                        woa_pos = population[rand_idx, j] - A_vec * D
                else:
                    # Phase 2: Spiral/Bubble-net attack (Eq 11)
                    D_prime = abs(gbest[j] - curr)
                    woa_pos = D_prime * np.exp(b_spiral * l_param) * np.cos(2 * np.pi * l_param) + gbest[j]

                # Clamp WOA displacement to prevent large jumps
                displacement = woa_pos - curr
                disp_norm = np.linalg.norm(displacement)
                if disp_norm > step_limit:
                    woa_pos = curr + displacement * (step_limit / disp_norm)

                # Apply APF force (attractive + repulsive) with scaling
                apf_force = compute_apf_force(woa_pos)
                woa_pos = woa_pos + apf_force * APF_SCALE

                # Smoothing: move toward neighbors' midpoint
                if 0 < j < N_CTRL - 1:
                    midpoint = (new_ctrl[j-1] + population[i, j+1]) / 2.0
                    woa_pos = woa_pos + 0.3 * (midpoint - woa_pos)

                # Clamp to grid and accept if collision-free
                woa_pos = np.clip(woa_pos, 0, GRID_SIZE - 1)
                gy = int(np.clip(round(woa_pos[0]), 0, GRID_SIZE - 1))
                gx = int(np.clip(round(woa_pos[1]), 0, GRID_SIZE - 1))
                if grid[gy, gx] == 0:
                    new_ctrl[j] = woa_pos

            # Greedy acceptance: only accept new position set if it improves cost
            new_cost = evaluate_whale(new_ctrl)
            if new_cost < whale_costs[i]:
                population[i] = new_ctrl
                whale_costs[i] = new_cost
                if new_cost < gbest_cost:
                    gbest_cost = new_cost
                    gbest = new_ctrl.copy()

        # Random jump mechanism: escape local minima (low probability)
        for i in range(N_WHALES):
            if np.random.rand() < 0.05:
                trial = np.copy(population[i])
                rand_j = np.random.randint(0, N_CTRL)
                trial[rand_j] += np.random.uniform(-2, 2, 2)
                trial[rand_j] = np.clip(trial[rand_j], 0, GRID_SIZE - 1)
                trial_cost = evaluate_whale(trial)
                if trial_cost < whale_costs[i]:
                    population[i] = trial
                    whale_costs[i] = trial_cost
                    if trial_cost < gbest_cost:
                        gbest_cost = trial_cost
                        gbest = trial.copy()

    # Return best whale's control points
    sorted_pts = gbest[np.argsort(np.linalg.norm(gbest - start_arr, axis=1))]
    full_pts = np.vstack([start_arr, sorted_pts, goal_arr])
    return sparsify_path(np.array(full_pts, dtype=float), min_dist=5.0)

def run_ab_woa(start_node, goal_node, start_dir=None, goal_dir=None):
    inflated_grid = np.zeros_like(grid)
    inflated_grid[dist_field < 2] = 1 
    obstacles_list = np.argwhere(grid == 1)
    sparse_path = solve_bio_hybrid_competitor(grid, inflated_grid, obstacles_list, start_node, goal_node)
    
    # Add start direction tangent for kinematic consistency
    start_pts_list = [np.array(start_node)]
    if start_dir is not None:
        start_pts_list.append(_safe_tangent_point(start_node, start_dir))
    # Add goal direction tangent so robot approaches goal heading toward next WP
    if goal_dir is not None:
        gt = _safe_tangent_point(goal_node, -np.array(goal_dir))
        full_ctrl = np.vstack([start_pts_list, sparse_path[1:-1], [gt], [np.array(goal_node, dtype=float)]])
    else:
        full_ctrl = np.vstack([start_pts_list, sparse_path[1:]])
    final_path = common_spline(full_ctrl, PLOT_SAMPLES)
    final_path = ensure_collision_free(final_path)
    # Pin endpoints to exact waypoints
    final_path[0] = np.array(start_node, dtype=float)
    final_path[-1] = np.array(goal_node, dtype=float)
    
    if len(final_path) > 50:
        end_dir = final_path[-1] - final_path[-50]
        norm = np.linalg.norm(end_dir)
        end_dir = end_dir / norm if norm > 0 else np.array([0,1])
    else: end_dir = np.array([0,1])
    return final_path, end_dir

# --- 3. HWPSO (Paper: "Enhanced path planning algorithm via hybrid WOA-PSO") ---
# Eq 15 (hybrid velocity): V = w·V + c1·r1·(Whale* - X_woa) + c2·r2·(gbest - X_woa)
# Eq 20 (cost): f(·) = f_p · (1 + μ · v_L)
# WOA phases (Eq 10,11,14) use gbest as X*; PSO cognitive term uses Whale_Star (Eq 15)
def run_woa_pso(start_node, goal_node, start_dir=None, goal_dir=None):
    POP_SIZE, MAX_ITER, DIM = 40, 30, 5 
    X = np.zeros((POP_SIZE, DIM, 2))
    V = np.zeros((POP_SIZE, DIM, 2))
    
    # Initialization
    for i in range(POP_SIZE):
        for j in range(DIM):
            alpha = (j + 1) / (DIM + 1)
            y = (1-alpha)*start_node[0] + alpha*goal_node[0] + np.random.uniform(-5,5)
            x = (1-alpha)*start_node[1] + alpha*goal_node[1] + np.random.uniform(-5,5)
            X[i, j] = [np.clip(y,0,GRID_SIZE-1), np.clip(x,0,GRID_SIZE-1)]

    gbest = X[0].copy(); gbest_cost = float('inf')
    
    start_pts_list = [np.array(start_node)]
    if start_dir is not None:
        start_pts_list.append(_safe_tangent_point(start_node, start_dir))
    start_pts = np.array(start_pts_list)
    # Goal tangent points for heading continuity
    goal_pts_list = []
    if goal_dir is not None:
        goal_pts_list.append(_safe_tangent_point(goal_node, -np.array(goal_dir)))
    goal_pts_list.append(np.array(goal_node, dtype=float))
    goal_pts = np.array(goal_pts_list)

    def evaluate(ind):
        pts = ind[np.argsort(np.linalg.norm(ind - np.array(start_node), axis=1))]
        full_pts = np.vstack([start_pts, pts, goal_pts])
        path = common_spline(full_pts, CHECK_SAMPLES)
        f_p = get_path_length(path)
        v_L = calculate_violation_paper(path) # Uses SUM (tuned from paper's MEAN, Eq 18)
        
        # Eq 20: cost = f_p * (1 + μ * v_L). Paper μ=100; tuned to 10000 for safety.
        MU = 10000.0 
        cost = f_p * (1.0 + MU * v_L)
        
        cost += get_curvature_penalty(path) + get_curvature_integral_penalty(path)
        if start_dir is not None: cost += check_forward_motion(path, start_dir)
        return cost

    w = 0.6; c1 = 1.2; c2 = 1.2; V_MAX = 2.0 

    for t in range(MAX_ITER):
        a = 2 - 2 * t / MAX_ITER  # WOA: a decreases from 2 to 0
        current_iter_best = None; current_iter_best_cost = float('inf')
        for i in range(POP_SIZE):
            cost = evaluate(X[i])
            if cost < current_iter_best_cost:
                current_iter_best_cost = cost; current_iter_best = X[i].copy()
            if cost < gbest_cost:
                gbest_cost = cost; gbest = X[i].copy()
        
        # Whale_Star = best of current iteration (used as Whale* in Eq 15, replacing pbest)
        Whale_Star = current_iter_best if current_iter_best is not None else gbest

        for i in range(POP_SIZE):
            r1 = np.random.rand(); r2 = np.random.rand()
            A = 2*a*r1 - a; C = 2*r2  # Eq 9: A = 2a·r-a, C = 2r
            p = np.random.rand(); l = np.random.uniform(-1, 1); b = 1
            X_woa = X[i].copy()
            if p < 0.5:
                # WOA Eq 10 & 14: X* = gbest (global best found), NOT iteration best
                if abs(A) < 1: D = abs(C * gbest - X[i]); X_woa = gbest - A * D           # Encircling (Eq 10)
                else: rand_idx = np.random.randint(0, POP_SIZE); D = abs(C * X[rand_idx] - X[i]); X_woa = X[rand_idx] - A * D  # Exploration (Eq 14)
            else: D_prime = abs(gbest - X[i]); X_woa = D_prime * np.exp(b*l) * np.cos(2*np.pi*l) + gbest  # Spiral (Eq 11)
            
            # Eq 15 (HWPSO hybrid): Whale_Star replaces pbest in PSO cognitive term
            r1_pso = np.random.rand(); r2_pso = np.random.rand()
            V[i] = w * V[i] + c1 * r1_pso * (Whale_Star - X_woa) + c2 * r2_pso * (gbest - X_woa)
            V[i] = np.clip(V[i], -V_MAX, V_MAX)
            X[i] = np.clip(X_woa + V[i], 0, GRID_SIZE-1)

    pts = gbest[np.argsort(np.linalg.norm(gbest - np.array(start_node), axis=1))]
    final_path = common_spline(np.vstack([start_pts, pts, goal_pts]), PLOT_SAMPLES)
    final_path = ensure_collision_free(final_path)
    # Pin endpoints to exact waypoints
    final_path[0] = np.array(start_node, dtype=float)
    final_path[-1] = np.array(goal_node, dtype=float)
    end_dir = final_path[-1] - final_path[-50]
    norm = np.linalg.norm(end_dir)
    end_dir = end_dir / norm if norm > 0 else np.array([0,1])
    return final_path, end_dir

# --- 5. NSGA-II (OPTIMIZED) ---
def run_nsga_ii(start_node, goal_node, start_dir=None, goal_dir=None, **kwargs):
    # Tunable hyperparameters (defaults = production values)
    pop_size   = kwargs.get('pop_size', 60)
    ngen       = kwargs.get('ngen', 40)
    cxpb       = kwargs.get('cxpb', 0.7)
    mutpb      = kwargs.get('mutpb', 0.3)
    mut_sigma  = kwargs.get('mut_sigma', 2.0)
    indpb      = kwargs.get('indpb', 0.2)
    seed_ratio = kwargs.get('seed_ratio', 0.7)   # fraction of pop seeded from A*
    n_ctrl_pts = kwargs.get('n_ctrl_pts', 5)      # number of interior control points

    raw_path = a_star(grid, start_node, goal_node)
    if raw_path and len(raw_path) > (n_ctrl_pts + 3):
        indices = np.linspace(0, len(raw_path)-1, n_ctrl_pts + 2, dtype=int)
        raw_seed = np.array(raw_path)[indices][1:-1].astype(float)
        seed_pts = [tuple(pt) for pt in raw_seed] if len(raw_seed) >= 2 else [tuple(start_node), tuple(goal_node)]
    else:
        seed_pts = [tuple(start_node), tuple(goal_node)]

    start_pts_fixed = [np.array(start_node)]
    if start_dir is not None:
        start_pts_fixed.append(_safe_tangent_point(start_node, start_dir))

    # Goal tangent: approach the goal heading toward the NEXT waypoint (goal_dir)
    # This ensures non-holonomic heading continuity at waypoint transitions.
    # If goal_dir is provided, the robot arrives at the goal already facing
    # the next waypoint, avoiding impossible U-turns for a two-wheeled robot.
    # Uses _safe_tangent_point to avoid placing tangent inside obstacles.
    if goal_dir is not None:
        approach_dir = np.array(goal_dir, dtype=float)
    else:
        # Fallback: approach from start→goal direction
        sg_vec = np.array(goal_node) - np.array(start_node)
        sg_norm = np.linalg.norm(sg_vec)
        approach_dir = sg_vec / sg_norm if sg_norm > 1e-6 else np.array([0.0, 1.0])
    goal_tangent = _safe_tangent_point(goal_node, -approach_dir)
    goal_pts_fixed = [goal_tangent, np.array(goal_node)]

    if hasattr(creator, "FitnessNSGA"): del creator.FitnessNSGA
    if hasattr(creator, "IndividualNSGA"): del creator.IndividualNSGA
    creator.create("FitnessNSGA", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0))
    creator.create("IndividualNSGA", list, fitness=creator.FitnessNSGA)

    obstacles_list = np.argwhere(grid == 1)

    def eval_nsga(ind):
        full_pts = start_pts_fixed + list(ind) + goal_pts_fixed
        path = common_spline(full_pts, CHECK_SAMPLES)
        length = get_path_length(path)
        
        # Kinematic: both max-curvature and integral penalty
        max_k_penalty = get_curvature_penalty(path) + get_curvature_integral_penalty(path)
        
        effort = calculate_wheel_effort(path)
        center = calculate_centering_score(path, obstacles_list)
        
        # --- Graduated Safety (replaces death penalty) ---
        safety_pen = 0.0
        min_x, min_y = np.min(path[:, 1]), np.min(path[:, 0])
        max_x, max_y = np.max(path[:, 1]), np.max(path[:, 0])
        if min_x < 0.5 or min_y < 0.5 or max_x > GRID_SIZE-0.5 or max_y > GRID_SIZE-0.5:
            safety_pen += 500.0
        
        iy = np.round(path[:, 0]).astype(int); ix = np.round(path[:, 1]).astype(int)
        iy = np.clip(iy, 0, GRID_SIZE - 1); ix = np.clip(ix, 0, GRID_SIZE - 1)
        dists = dist_field[iy, ix] * GRID_RES
        unsafe_count = np.sum(dists < MIN_SAFE_DIST)
        if unsafe_count > 0:
            safety_pen += unsafe_count * 50.0  # Graduated penalty
        else:
            # For safe paths, proximity measure (lower = further from obstacles)
            safety_pen = np.sum(1.0 / (dists + 0.1)) * 0.01
        
        if start_dir is not None:
            safety_pen += check_forward_motion(path, start_dir)
        
        return (length, max_k_penalty, effort, center, safety_pen)

    def mut_nsga(ind, _indpb=None):
        _indpb = _indpb if _indpb is not None else indpb
        for i in range(len(ind)):
            if random.random() < _indpb:
                ind[i] = (np.clip(ind[i][0] + random.gauss(0, mut_sigma), 0, GRID_SIZE-1),
                          np.clip(ind[i][1] + random.gauss(0, mut_sigma), 0, GRID_SIZE-1))
        return ind,

    toolbox = base.Toolbox()
    n_ctrl = len(seed_pts)
    def init_nsga_ind():
        if random.random() < (1.0 - seed_ratio):
            # Random straight-line paths with noise
            pts = []
            for j in range(n_ctrl):
                alpha = (j + 1) / (n_ctrl + 1)
                y = (1-alpha)*start_node[0] + alpha*goal_node[0] + random.uniform(-5, 5)
                x = (1-alpha)*start_node[1] + alpha*goal_node[1] + random.uniform(-5, 5)
                pts.append((np.clip(y, 0, GRID_SIZE-1), np.clip(x, 0, GRID_SIZE-1)))
        else:
            # Seeded from A* with random perturbation
            pts = []
            for pt in seed_pts:
                ny = np.clip(pt[0] + random.gauss(0, 2.5), 0, GRID_SIZE-1)
                nx = np.clip(pt[1] + random.gauss(0, 2.5), 0, GRID_SIZE-1)
                pts.append((ny, nx))
        return creator.IndividualNSGA(pts)
    toolbox.register("individual", init_nsga_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_nsga)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mut_nsga, _indpb=indpb)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    # DEAP requires cxpb + mutpb <= 1.0; clamp if sensitivity sweep pushes over
    if cxpb + mutpb > 1.0:
        total = cxpb + mutpb
        cxpb = cxpb / total
        mutpb = mutpb / total
    algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                              cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)
    
    best_len = min(pop, key=lambda x: x.fitness.values[0])
    best_curv = min(pop, key=lambda x: x.fitness.values[1])
    best_effort = min(pop, key=lambda x: x.fitness.values[2])
    best_center = min(pop, key=lambda x: x.fitness.values[3])
    best_safe = min(pop, key=lambda x: x.fitness.values[4])
    
    champions = [
        start_pts_fixed + list(best_len) + goal_pts_fixed,
        start_pts_fixed + list(best_curv) + goal_pts_fixed,
        start_pts_fixed + list(best_effort) + goal_pts_fixed,
        start_pts_fixed + list(best_center) + goal_pts_fixed,
        start_pts_fixed + list(best_safe) + goal_pts_fixed
    ]
    
    adaptive_coords = construct_adaptive_path(champions, obstacles_list)
    # adaptive_coords is ~100 smooth control points; use approximating spline
    # (smoothing > 0) to suppress residual noise and prevent cubic overshoot
    if len(adaptive_coords) > 2:
        path_adaptive = ensure_collision_free(
            common_spline(adaptive_coords, PLOT_SAMPLES,
                          smoothing=len(adaptive_coords) * 0.5))
    else:
        path_adaptive = ensure_collision_free(common_spline(champions[0], PLOT_SAMPLES))

    # Multi-pass kinematic pipeline:
    # 1. Vectorized smoothing (fast convergence, 500 iters each)
    # 2. Direct curvature enforcement to fix remaining violations
    # 3. Final re-spline to guarantee smooth output
    for _ in range(3):
        path_adaptive = smooth_path_kinematic(path_adaptive, iterations=500,
                                              alpha=0.25, beta=0.60)
        path_adaptive = enforce_max_curvature(path_adaptive, max_iters=30)
    path_adaptive = curvature_limit_respline(path_adaptive)
    path_adaptive = _escape_obstacles(path_adaptive)

    paths = {}
    for key, champ in zip(["Length", "Smooth", "Effort", "Centered", "Safe"], champions):
        p = ensure_collision_free(common_spline(champ, PLOT_SAMPLES))
        p[0] = np.array(start_node, dtype=float)
        p[-1] = np.array(goal_node, dtype=float)
        paths[key] = p
    path_adaptive[0] = np.array(start_node, dtype=float)
    path_adaptive[-1] = np.array(goal_node, dtype=float)
    paths["Adaptive"] = path_adaptive
    
    end_dir = path_adaptive[-1] - path_adaptive[-50]
    norm = np.linalg.norm(end_dir)
    end_dir = end_dir / norm if norm > 0 else np.array([0,1])
    return paths, end_dir

# ---------------------------------------------------------
# 6. EXECUTION & PLOTTING
# ---------------------------------------------------------
def create_patrol_environment(difficulty):
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    waypoints = [] 

    # --- EXISTING ENVIRONMENTS ---
    '''if difficulty == "Easy":
        g[2:8, 22:28] = 1; g[42:48, 22:28] = 1 
        waypoints = [(5, 5), (5, 45), (45, 45), (45, 5)]
    elif difficulty == "Moderate":
        g[2:8, 22:28] = 1; g[22:28, 42:48] = 1 
        g[42:48, 22:28] = 1; g[22:28, 2:8] = 1   
        waypoints = [(5, 5), (5, 45), (45, 45), (45, 5)]'''
    if difficulty == "Easy":
        g[0:10, 20:30] = 1; g[20:30, 40:50] = 1  
        g[40:50, 20:30] = 1; g[20:30, 0:10] = 1   
        g[22:28, 22:28] = 1  
        waypoints = [(5, 5), (5, 45), (45, 45), (45, 5)]
    elif difficulty == "Moderate-I":
        waypoints = [(5, 5), (45, 15), (5, 25), (45, 35), (5, 45)]
        g[20:35, 8:12] = 1; g[15:30, 18:22] = 1  
        g[20:35, 28:32] = 1; g[15:30, 38:42] = 1  
        g[10:40, 2:6] = 1 
    elif difficulty == "Moderate-II":
        g[10:20, 10:15] = 1; g[35:45, 10:15] = 1; g[20:30, 22:28] = 1; g[2:10, 22:28] = 1
        g[10:20, 35:40] = 1; g[35:45, 35:40] = 1; g[22:25, 0:10] = 1; g[22:25, 40:50] = 1; g[45:48, 20:30] = 1
        waypoints = [(5, 5), (45, 5), (45, 45), (5, 45)]

    # --- UPDATED: LITERATURE_MIXED (More Obstacles) ---
    elif difficulty == "Moderate-III":
        # 1. CIRCLE (Top Right)
        cx, cy, r = 35, 35, 6
        y, x = np.ogrid[:GRID_SIZE, :GRID_SIZE]
        mask_circle = (x - cx)**2 + (y - cy)**2 <= r**2
        g[mask_circle] = 1
        
        # 2. SMALL CIRCLE (Top Left) - NEW
        cx2, cy2, r2 = 10, 40, 4
        mask_circle2 = (x - cx2)**2 + (y - cy2)**2 <= r2**2
        g[mask_circle2] = 1

        # 3. TRIANGLE (Bottom Center)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (10 <= i <= 20) and (25 - (i-10) <= j <= 25 + (i-10)):
                    g[i, j] = 1

        # 4. RECTANGLE (Left Side)
        g[15:35, 5:10] = 1

        # 5. CROSS (Center)
        g[22:28, 24:26] = 1; g[24:26, 22:28] = 1 

        # 6. DIAGONAL BAR (Bottom Right)
        for k in range(8):
            g[6+k, 37+k] = 1; g[6+k, 38+k] = 1

        # 7. DIAMOND/RHOMBUS (Right Center) - NEW
        # Center (30, 42), radius approx 4
        for i in range(25, 35):
            for j in range(38, 46):
                if abs(i - 30) + abs(j - 42) <= 4:
                    g[i, j] = 1
        
        # 8. L-SHAPE (Bottom Left) - NEW
        g[5:12, 12:15] = 1  # Vertical part
        g[5:8, 12:18] = 1   # Horizontal part

        # 9. SCATTERED NOISE (Small 2x2 blocks) - NEW
        g[44:46, 20:22] = 1
        g[40:42, 28:30] = 1
        g[15:17, 45:47] = 1

        waypoints = [
            (5, 5),    # Bottom Left
            (45, 15),  # Top Left
            (25, 40),  # Top Center (Moved DOWN from 45 to 40 to prevent loop/crash)
            (5, 35)    # Bottom Right
        ]

    # --- NEW: COMPLEX WAREHOUSE ENVIRONMENT ---
    # Challenging benchmark designed for Q1/Q2 journal publication.
    # Combines multiple difficulty factors that stress-test path planners:
    #   (1) Narrow aisles between shelf units (7-10 cell gaps)
    #   (2) Central partition wall with narrow doorway (bottleneck)
    #   (3) U-shaped & C-shaped concave obstacles (dead-end traps)
    #   (4) Circular structural pillars with irregular clearances
    #   (5) Diagonal barriers blocking direct-line shortcuts
    #   (6) Six patrol waypoints across distinct zones, forcing
    #       multiple crossings of the central partition
    #   (7) Asymmetric layout preventing symmetry exploitation
    #
    # The combination of R_min=2.5 m turning constraint, narrow corridors,
    # and concave traps creates a severe stress test for non-holonomic
    # multi-waypoint patrol planning. Greedy/reactive planners tend to
    # fail near the U-trap and central bottleneck, while single-objective
    # optimizers struggle to balance safety vs. length in the tight aisles.
    elif difficulty == "Hard":

        # --- (1) SHELF UNITS: Three rows on left, two on right ---
        # Creates narrow aisles the robot must navigate through.
        # Aisle width ≈ 7-10 cells (0.7-1.0 m) — passable but demanding.

        # Left-side shelves (three rows)
        g[8:14, 3:7] = 1       # Shelf A1  (bottom-left)
        g[8:14, 14:18] = 1     # Shelf A2
        g[22:28, 3:7] = 1      # Shelf B1  (mid-left)
        g[22:28, 14:18] = 1    # Shelf B2
        g[36:42, 3:7] = 1      # Shelf C1  (top-left)
        g[36:42, 14:18] = 1    # Shelf C2

        # Right-side shelves (two rows, offset for asymmetry)
        g[10:16, 35:39] = 1    # Shelf D1  (bottom-right)
        g[10:16, 43:47] = 1    # Shelf D2
        g[28:34, 35:39] = 1    # Shelf E1  (mid-right)
        g[28:34, 43:47] = 1    # Shelf E2

        # --- (2) CENTRAL PARTITION WALL with narrow doorway ---
        # A vertical wall that divides the map into west/east halves.
        # Doorway gap: rows 22-28 (7 cells ≈ 0.7 m effective width).
        # Center clearance = 3.5 cells (0.35 m) > MIN_SAFE_DIST (0.3 m).
        g[10:22, 24:27] = 1    # Lower wall section
        g[29:40, 24:27] = 1    # Upper wall section

        # --- (3) U-SHAPED OBSTACLE (bottom centre) ---
        # Creates a concave trap: greedy planners may enter and fail
        # to exit without violating curvature constraints.
        g[2:8, 20:22] = 1      # Left arm
        g[2:8, 30:32] = 1      # Right arm
        g[2:4, 20:32] = 1      # Bottom connecting bar

        # --- (4) C-SHAPED OBSTACLE (top right) ---
        # Another concave trap on the opposite side of the map.
        g[42:48, 32:34] = 1    # Left arm
        g[42:48, 42:44] = 1    # Right arm
        g[46:48, 32:44] = 1    # Top connecting bar

        # --- (5) CIRCULAR STRUCTURAL PILLARS ---
        y, x = np.ogrid[:GRID_SIZE, :GRID_SIZE]
        for (cy, cx, r) in [
            (18, 10, 2),   # Lower-left pillar
            (18, 40, 2),   # Lower-right pillar
            (32, 10, 2),   # Upper-left pillar
            (32, 40, 2),   # Upper-right pillar
        ]:
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            g[mask] = 1

        # --- (6) DIAGONAL BARRIERS (block direct-line shortcuts) ---
        # Lower-right diagonal
        for k in range(8):
            yi, xi = 5 + k, 38 + k
            if yi < GRID_SIZE and xi < GRID_SIZE:
                g[yi:min(yi + 2, GRID_SIZE), xi:min(xi + 2, GRID_SIZE)] = 1
        # Upper-left diagonal
        for k in range(6):
            yi, xi = 38 + k, 8 + k
            if yi < GRID_SIZE and xi < GRID_SIZE:
                g[yi:min(yi + 2, GRID_SIZE), xi:min(xi + 2, GRID_SIZE)] = 1

        # --- (7) SCATTERED CLUTTER (small blocks adding noise) ---
        g[20:22, 42:44] = 1
        g[40:42, 28:30] = 1
        g[4:6, 10:12] = 1
        g[44:46, 20:22] = 1
        g[15:17, 30:32] = 1

        # --- EIGHT PATROL WAYPOINTS (interior + edge) ---
        # Forces the robot through every constrained mid-section:
        #   • Shelf aisles (WP2 between A-row shelves, WP8 between B/C shelves)
        #   • Central doorway bottleneck (WP4)
        #   • Near U-shape trap (WP3 sits just above the U opening)
        #   • Inside C-shape concavity (WP6 — must enter AND exit)
        #   • Close to pillars (WP2 near pillar@18,10; WP5 near pillar@18,40;
        #     WP8 near pillar@32,10)
        # The loop 1→2→3→4→5→6→7→8→1 requires ≥4 crossings of the
        # central partition and navigation through every tight zone.
        waypoints = [
            (2, 2),       # WP1: Bottom-left corner (open start area)
            (12, 10),     # WP2: Left shelf aisle interior (between A1/A2, near pillar)
            (8, 26),      # WP3: Bottom centre, just above U-shape trap opening
            (25, 25),     # WP4: Central doorway bottleneck (partition crossing)
            (25, 41),     # WP5: Right-side interior (between shelf D rows, near pillar)
            (44, 38),     # WP6: Inside C-shape concavity (must enter & exit trap)
            (47, 5),      # WP7: Top-left corner (past upper diagonal barrier)
            (32, 15),     # WP8: Upper-left shelf aisle (between B/C shelves, near pillar)
        ]

    # --- SAFETY CHECK ---
    final_waypoints = []
    for wy, wx in waypoints:
        if g[int(wy), int(wx)] == 1:
            print(f"Warning: Waypoint ({wy},{wx}) on obstacle! Shifting...")
            found = False
            for r in range(1, 8): 
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        ny, nx = np.clip(wy+dy, 0, GRID_SIZE-1), np.clip(wx+dx, 0, GRID_SIZE-1)
                        if g[ny, nx] == 0:
                            final_waypoints.append((ny, nx))
                            found = True; break
                    if found: break
                if found: break
            if not found: final_waypoints.append((wy, wx))
        else:
            final_waypoints.append((wy, wx))

    return g, final_waypoints

def solve_patrol_single(algo_func, waypoints):
    full_path = []
    targets = waypoints.copy()
    targets.append(waypoints[0]) 
    curr_dir = FIXED_START_DIR.copy()
    for i in range(len(targets) - 1):
        start, end = targets[i], targets[i+1]
        # Compute goal_dir: direction from 'end' toward the NEXT waypoint
        # This ensures the robot arrives at 'end' heading toward the
        # subsequent waypoint, maintaining non-holonomic feasibility.
        if i + 2 < len(targets):
            next_after = np.array(targets[i + 2], dtype=float)
        else:
            # Last segment returns to WP1; next target in cycle is WP2
            next_after = np.array(targets[1], dtype=float)
        gd_vec = next_after - np.array(end, dtype=float)
        gd_norm = np.linalg.norm(gd_vec)
        goal_dir = gd_vec / gd_norm if gd_norm > 1e-6 else curr_dir
        segment, end_dir = algo_func(start, end, start_dir=curr_dir, goal_dir=goal_dir)
        # Guarantee segment endpoints match waypoints exactly
        segment[0] = np.array(start, dtype=float)
        segment[-1] = np.array(end, dtype=float)
        # Use the planned arrival heading (goal_dir) as next segment's departure
        # direction, not the computed end_dir which may diverge from goal_dir
        # when tangent is weak or post-processing alters the path end.
        curr_dir = goal_dir
        if len(full_path) == 0: full_path = segment
        else: full_path = np.vstack((full_path, segment[1:]))
    return full_path

def solve_patrol_nsga_all(waypoints, **nsga_kwargs):
    full_paths = {k: [] for k in ["Length", "Smooth", "Effort", "Centered", "Safe", "Adaptive"]}
    targets = waypoints.copy()
    targets.append(waypoints[0])
    curr_dir = FIXED_START_DIR.copy()
    # Track junction indices for post-smoothing
    junction_indices = {k: [] for k in full_paths}
    for i in range(len(targets) - 1):
        start, end = targets[i], targets[i+1]
        # Compute goal_dir: direction from 'end' toward the NEXT waypoint
        # Ensures non-holonomic heading continuity at waypoint transitions.
        if i + 2 < len(targets):
            next_after = np.array(targets[i + 2], dtype=float)
        else:
            next_after = np.array(targets[1], dtype=float)
        gd_vec = next_after - np.array(end, dtype=float)
        gd_norm = np.linalg.norm(gd_vec)
        goal_dir = gd_vec / gd_norm if gd_norm > 1e-6 else curr_dir
        seg_dict, end_dir = run_nsga_ii(start, end, start_dir=curr_dir, goal_dir=goal_dir, **nsga_kwargs)
        # Use the planned arrival heading as next departure direction
        curr_dir = goal_dir
        for k in full_paths:
            # Guarantee segment endpoints match waypoints exactly
            seg_dict[k][0] = np.array(start, dtype=float)
            seg_dict[k][-1] = np.array(end, dtype=float)
            if len(full_paths[k]) == 0:
                full_paths[k] = seg_dict[k]
            else:
                junction_indices[k].append(len(full_paths[k]) - 1)
                full_paths[k] = np.vstack((full_paths[k], seg_dict[k][1:]))

    # Post-concatenation junction smoothing for the Adaptive path
    # This eliminates sharp turns where segments meet at waypoints
    for k in ["Adaptive"]:
        path = full_paths[k]
        if len(path) < 10:
            continue
        for jidx in junction_indices[k]:
            # Smooth a local window around each junction
            window = min(80, len(path) // 8)  # ~80 points each side
            lo = max(1, jidx - window)
            hi = min(len(path) - 2, jidx + window)
            if hi - lo < 4:
                continue
            # Local iterative smoothing
            safe_threshold = MIN_SAFE_DIST / GRID_RES
            for _ in range(300):
                new_seg = path[lo:hi+1].copy()
                for j in range(1, len(new_seg) - 1):
                    avg = (new_seg[j-1] + new_seg[j+1]) / 2.0
                    cand = new_seg[j] + 0.5 * (avg - new_seg[j])
                    cand[0] = np.clip(cand[0], 0, GRID_SIZE - 1)
                    cand[1] = np.clip(cand[1], 0, GRID_SIZE - 1)
                    cy = int(np.clip(round(cand[0]), 0, GRID_SIZE - 1))
                    cx = int(np.clip(round(cand[1]), 0, GRID_SIZE - 1))
                    if dist_field[cy, cx] >= safe_threshold:
                        new_seg[j] = cand
                path[lo:hi+1] = new_seg
        # Apply global curvature enforcement after junction smoothing
        full_paths[k] = enforce_max_curvature(path, max_iters=30)
    return full_paths

def plot_boxplot(all_distances, diff, algo_names, algo_colors):
    """Standard boxplot for path distances across 100 epochs."""
    fig, ax = plt.subplots(figsize=(14, 7))
    data = [all_distances[name] for name in algo_names]
    
    bp = ax.boxplot(data, patch_artist=True, notch=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    for patch, color in zip(bp['boxes'], [algo_colors[n] for n in algo_names]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticklabels(algo_names, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Path Distance (m)', fontsize=12)
    #ax.set_title(f'Path Distance Distribution — {diff} (100 epochs, no fixed seed)', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(f"Boxplot_{diff}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved Boxplot_{diff}.png")


def plot_raincloud(all_distances, diff, algo_names, algo_colors):
    """Raincloud plot: half-violin + jittered scatter + boxplot."""
    fig, ax = plt.subplots(figsize=(14, 7))
    data = [np.array(all_distances[name]) for name in algo_names]
    n_algos = len(algo_names)
    positions = np.arange(1, n_algos + 1)

    # --- Half-violin (rain cloud) ---
    vp = ax.violinplot(data, positions=positions, showmeans=False,
                       showmedians=False, showextrema=False, widths=0.7)
    for i, body in enumerate(vp['bodies']):
        # Keep only right half of the violin
        m = np.mean(body.get_paths()[0].vertices[:, 0])
        body.get_paths()[0].vertices[:, 0] = np.clip(
            body.get_paths()[0].vertices[:, 0], m, np.inf)
        body.set_facecolor(algo_colors[algo_names[i]])
        body.set_edgecolor('black')
        body.set_linewidth(0.8)
        body.set_alpha(0.6)

    # --- Jittered scatter (rain drops) on the left side ---
    for i, d in enumerate(data):
        jitter = np.random.uniform(-0.15, 0.0, size=len(d))
        ax.scatter(positions[i] + jitter, d, s=12, alpha=0.4,
                   color=algo_colors[algo_names[i]], edgecolors='none', zorder=3)

    # --- Boxplot (compact, centered) ---
    bp = ax.boxplot(data, positions=positions, widths=0.15, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    boxprops=dict(linewidth=1.0))
    for patch, name in zip(bp['boxes'], algo_names):
        patch.set_facecolor(algo_colors[name])
        patch.set_alpha(0.9)

    ax.set_xticks(positions)
    ax.set_xticklabels(algo_names, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Path Distance (m)', fontsize=12)
    #ax.set_title(f'Raincloud — Path Distance Distribution — {diff} (100 epochs, no fixed seed)', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(f"Raincloud_{diff}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved Raincloud_{diff}.png")


# ---------------------------------------------------------
# MODE 1: SINGLE EPOCH — path figures (like original)
# ---------------------------------------------------------
def run_single_epoch_mode():
    """Run 1 epoch per environment with a fixed seed, produce path overlay figures."""
    seedno = 50
    random.seed(seedno); np.random.seed(seedno)
    global grid, dist_field
    ENV_DIFFICULTIES = ["Easy", "Moderate-I", "Moderate-II", "Moderate-III", "Hard"]
    timing_results = {}

    for diff in ENV_DIFFICULTIES:
        print(f"Running Environment: {diff}")
        grid, waypoints = create_patrol_environment(diff)
        dist_field = ndi.distance_transform_edt(1 - grid)

        # 1. Run Competitors
        t0 = time.time()
        p_ga = solve_patrol_single(run_ga_standard, waypoints)
        timing_results["Standard GA"] = round(time.time() - t0, 4)

        t0 = time.time()
        p_ab = solve_patrol_single(run_ab_woa, waypoints)
        timing_results["AB-WOA-APF"] = round(time.time() - t0, 4)

        t0 = time.time()
        p_wp = solve_patrol_single(run_woa_pso, waypoints)
        timing_results["HWPSO"] = round(time.time() - t0, 4)

        # 2. Run NSGA-II
        t0 = time.time()
        p_nsga = solve_patrol_nsga_all(waypoints)
        nsga_total_time = round(time.time() - t0, 4)

        timing_results["NSGA-II Length"] = nsga_total_time
        timing_results["NSGA-II Smooth"] = nsga_total_time
        timing_results["NSGA-II Effort"] = nsga_total_time
        timing_results["NSGA-II Centered"] = nsga_total_time
        timing_results["NSGA-II Safe"] = nsga_total_time
        timing_results["NSGA-II Adaptive"] = nsga_total_time

        with open(f"calculation_times_single_{diff}.json", "w") as f:
            json.dump(timing_results, f, indent=4)
        print(f"Timing saved to calculation_times_single_{diff}.json")

        # --- Collision Verification ---
        def _count_col(p, name):
            iy = np.clip(np.round(p[:, 0]).astype(int), 0, GRID_SIZE - 1)
            ix = np.clip(np.round(p[:, 1]).astype(int), 0, GRID_SIZE - 1)
            obs_hit = int(np.sum(grid[iy, ix] == 1))
            if obs_hit > 0:
                print(f"  COLLISION: {name} penetrates obstacle at {obs_hit} points!")
            return obs_hit
        total_col = 0
        total_col += _count_col(p_ga, "Standard GA")
        total_col += _count_col(p_ab, "AB-WOA-APF")
        total_col += _count_col(p_wp, "HWPSO")
        for k in ["Length", "Smooth", "Effort", "Centered", "Safe", "Adaptive"]:
            total_col += _count_col(p_nsga[k], f"NSGA-II {k}")
        if total_col == 0:
            print(f"  All paths collision-free in {diff}!")

        # --- Path Length & Curvature Summary ---
        def _info(p, name):
            l = get_path_length(p)
            k = get_max_curvature(p)
            print(f"    {name:25s}  len={l:6.2f}m  max_k={k:.3f}")
        _info(p_ga, "Standard GA")
        _info(p_ab, "AB-WOA-APF")
        _info(p_wp, "HWPSO")
        for k in ["Length", "Smooth", "Effort", "Centered", "Safe", "Adaptive"]:
            _info(p_nsga[k], f"NSGA-II {k}")

        # --- Path overlay figure ---
        plt.figure(figsize=(12, 12))
        plt.imshow(grid, cmap='Greys', origin='lower',
                   extent=[0, GRID_SIZE, 0, GRID_SIZE])
        wx, wy = zip(*waypoints)
        plt.scatter(np.array(wy) + 0.5, np.array(wx) + 0.5,
                    c='red', s=150, marker='X', zorder=20)

        def plot_with_arrows(path, c, s, w, base_label):
            length = get_path_length(path)
            label = f"{base_label} ({length:.2f}m)"
            px = path[:, 1] + 0.5   # shift to cell-center coords
            py = path[:, 0] + 0.5
            plt.plot(px, py, color=c, linestyle=s, linewidth=w, label=label, alpha=0.8)
            arrow_indices = [int(len(path) * 0.125), int(len(path) * 0.375),
                             int(len(path) * 0.625), int(len(path) * 0.875)]
            for i in arrow_indices:
                dx = px[i + 5] - px[i]
                dy = py[i + 5] - py[i]
                plt.arrow(px[i], py[i], dx * 0.1, dy * 0.1, shape='full', lw=0,
                          length_includes_head=True, head_width=0.8, color=c, zorder=25)

        plot_with_arrows(p_nsga["Length"], 'blue', ':', 1.5, '1. NSGA-II Length')
        plot_with_arrows(p_nsga["Smooth"], 'lime', '--', 1.5, '2. NSGA-II Smooth')
        plot_with_arrows(p_nsga["Effort"], 'pink', '--', 1.5, '3. NSGA-II Effort')
        plot_with_arrows(p_nsga["Centered"], 'cyan', '--', 1.5, '4. NSGA-II Centered')
        plot_with_arrows(p_nsga["Safe"], 'magenta', '--', 1.5, '5. NSGA-II Safe')

        plot_with_arrows(p_ga, 'gray', '-', 2.5, '7. Standard GA')
        plot_with_arrows(p_ab, 'red', '-', 2.5, '8. AB-WOA-APF')
        plot_with_arrows(p_wp, 'purple', '-', 2.5, '9. HWPSO')

        plot_with_arrows(p_nsga["Adaptive"], 'gold', '-', 2.5, '6. NSGA-II Adaptive')

        plt.legend(loc='upper right', fontsize='small')
        #plt.title(f"Comparison of 9 Path Planning Strategies ({diff})")
        plt.xlim(0, GRID_SIZE)
        plt.ylim(0, GRID_SIZE)
        #plt.xlabel('x (grid cells)')
        #plt.ylabel('y (grid cells)')
        plt.tight_layout()
        plt.grid(color='black', linestyle='-')
        plt.savefig(f"Patrolling_9Paths_{diff}.png", dpi=300)
        plt.close()
        print(f"Saved Patrolling_9Paths_{diff}.png")

    print("\nSingle-epoch mode complete.")


# ---------------------------------------------------------
# MODE 2: 100-EPOCH TEST — boxplot & raincloud
# ---------------------------------------------------------
def _test_epoch_worker(args):
    """Worker for a single test-mode epoch. Runs ALL algorithms and returns results.
    Each worker sets up its own global state to avoid inter-process races."""
    diff, epoch_seed = args

    global grid, dist_field
    grid, waypoints = create_patrol_environment(diff)
    dist_field = ndi.distance_transform_edt(1 - grid)

    random.seed(epoch_seed)
    np.random.seed(epoch_seed)

    result = {}

    # --- Run Competitors ---
    t0 = time.time()
    p_ga = solve_patrol_single(run_ga_standard, waypoints)
    result["Standard GA"] = {"distance": get_path_length(p_ga), "time": time.time() - t0}

    t0 = time.time()
    p_ab = solve_patrol_single(run_ab_woa, waypoints)
    result["AB-WOA-APF"] = {"distance": get_path_length(p_ab), "time": time.time() - t0}

    t0 = time.time()
    p_wp = solve_patrol_single(run_woa_pso, waypoints)
    result["HWPSO"] = {"distance": get_path_length(p_wp), "time": time.time() - t0}

    # --- Run NSGA-II ---
    t0 = time.time()
    p_nsga = solve_patrol_nsga_all(waypoints)
    t_nsga = time.time() - t0

    for k in ["Length", "Smooth", "Effort", "Centered", "Safe", "Adaptive"]:
        result[f"NSGA-II {k}"] = {"distance": get_path_length(p_nsga[k]), "time": t_nsga}

    # Collision check
    def _count_col(p):
        iy = np.clip(np.round(p[:, 0]).astype(int), 0, GRID_SIZE - 1)
        ix = np.clip(np.round(p[:, 1]).astype(int), 0, GRID_SIZE - 1)
        return int(np.sum(grid[iy, ix] == 1))

    col_total = (_count_col(p_ga) + _count_col(p_ab) + _count_col(p_wp) +
                 sum(_count_col(p_nsga[k]) for k in ["Length", "Smooth", "Effort", "Centered", "Safe", "Adaptive"]))
    result["_collisions"] = col_total

    return result


def run_test_mode(num_epochs=100):
    """Run num_epochs per environment (no fixed seed), produce boxplot & raincloud.
    Uses multiprocessing to parallelize epochs across CPU cores."""
    global grid, dist_field

    n_workers = max(1, os.cpu_count() - 1)
    print(f"Detected {os.cpu_count()} CPU cores — using {n_workers} parallel workers")

    ENV_DIFFICULTIES = ["Easy", "Moderate-I", "Moderate-II", "Moderate-III", "Hard"]

    ALGO_NAMES = [
        "Standard GA", "AB-WOA-APF", "HWPSO",
        "NSGA-II Length", "NSGA-II Smooth", "NSGA-II Effort",
        "NSGA-II Centered", "NSGA-II Safe", "NSGA-II Adaptive"
    ]
    ALGO_COLORS = {
        "Standard GA": "gray",
        "AB-WOA-APF": "red",
        "HWPSO": "purple",
        "NSGA-II Length": "blue",
        "NSGA-II Smooth": "lime",
        "NSGA-II Effort": "pink",
        "NSGA-II Centered": "cyan",
        "NSGA-II Safe": "magenta",
        "NSGA-II Adaptive": "gold",
    }

    for diff in ENV_DIFFICULTIES:
        print(f"\n{'='*60}")
        print(f"Environment: {diff}  —  {num_epochs} epochs (no fixed seed)")
        print(f"{'='*60}")

        # Set global state for single-process fallback
        grid, waypoints = create_patrol_environment(diff)
        dist_field = ndi.distance_transform_edt(1 - grid)

        # Storage for distances across epochs
        all_distances = {name: [] for name in ALGO_NAMES}
        all_times = {name: [] for name in ALGO_NAMES}

        # Build tasks
        tasks = []
        for epoch in range(num_epochs):
            epoch_seed = hash(("test", diff, epoch)) % (2**31)
            tasks.append((diff, epoch_seed))

        # Execute in parallel
        epoch_results = [None] * num_epochs
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_epoch = {}
                for epoch, task_args in enumerate(tasks):
                    future = executor.submit(_test_epoch_worker, task_args)
                    future_to_epoch[future] = epoch

                completed = 0
                for future in as_completed(future_to_epoch):
                    epoch = future_to_epoch[future]
                    try:
                        epoch_results[epoch] = future.result()
                        completed += 1
                        col = epoch_results[epoch]["_collisions"]
                        status = "OK" if col == 0 else f"COLLISIONS={col}"
                        print(f"  Epoch {epoch+1}/{num_epochs} ... {status}  [{completed}/{num_epochs}]")
                    except Exception as e:
                        print(f"  Epoch {epoch+1}/{num_epochs} ... FAILED: {e}")
                        epoch_results[epoch] = None
        else:
            # Single-core fallback
            for epoch, task_args in enumerate(tasks):
                print(f"  Epoch {epoch+1}/{num_epochs} ...", end=" ", flush=True)
                result = _test_epoch_worker(task_args)
                epoch_results[epoch] = result
                status = "OK" if result["_collisions"] == 0 else f"COLLISIONS={result['_collisions']}"
                print(status)

        # Collect results
        for result in epoch_results:
            if result is None:
                continue
            for name in ALGO_NAMES:
                all_distances[name].append(result[name]["distance"])
                all_times[name].append(result[name]["time"])

        # --- Summary statistics ---
        print(f"\n  === Distance Summary for {diff} (meters) ===")
        print(f"  {'Algorithm':25s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}  {'Median':>8s}")
        summary = {}
        for name in ALGO_NAMES:
            arr = np.array(all_distances[name])
            summary[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "all": [float(v) for v in arr],
            }
            print(f"  {name:25s}  {np.mean(arr):8.2f}  {np.std(arr):8.2f}  "
                  f"{np.min(arr):8.2f}  {np.max(arr):8.2f}  {np.median(arr):8.2f}")

        # Save raw data as JSON
        with open(f"distances_{diff}.json", "w") as f:
            json.dump(summary, f, indent=4)
        print(f"  Saved distances_{diff}.json")

        # Save timing data
        timing_summary = {}
        for name in ALGO_NAMES:
            arr = np.array(all_times[name])
            timing_summary[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }
        with open(f"calculation_times_{diff}.json", "w") as f:
            json.dump(timing_summary, f, indent=4)
        print(f"  Saved calculation_times_{diff}.json")

        # --- Plots ---
        plot_boxplot(all_distances, diff, ALGO_NAMES, ALGO_COLORS)
        plot_raincloud(all_distances, diff, ALGO_NAMES, ALGO_COLORS)

    print("\nTest mode complete.")


# ---------------------------------------------------------
# MODE 3: SENSITIVITY ANALYSIS for A* Initialized NSGA-II
# ---------------------------------------------------------
def _evaluate_nsga_path(path):
    """Compute evaluation metrics for a single NSGA-II patrol path.
    Returns dict with distance, max_curvature, safety, and collision flag."""
    length = get_path_length(path)
    max_k = get_max_curvature(path)

    # Safety: minimum clearance along the path (metres)
    iy = np.clip(np.round(path[:, 0]).astype(int), 0, GRID_SIZE - 1)
    ix = np.clip(np.round(path[:, 1]).astype(int), 0, GRID_SIZE - 1)
    dists_m = dist_field[iy, ix] * GRID_RES
    min_clearance = float(np.min(dists_m))

    # Collision: actual obstacle cell penetration
    collision = int(np.sum(grid[iy, ix] == 1))

    return {
        "distance": length,
        "max_curvature": max_k,
        "min_clearance": min_clearance,
        "collision_pts": collision,
    }


def plot_sensitivity(results, param_name, param_values, env_name, metrics_to_plot=None):
    """No-op — all plotting handled by plot_sensitivity_all."""
    pass


def plot_sensitivity_combined(all_env_results, param_name, param_values,
                              env_names, metrics_to_plot=None):
    """No-op — all plotting handled by plot_sensitivity_all."""
    pass


def plot_sensitivity_all(all_param_results, param_sweeps, env_names,
                         metrics_to_plot=None):
    """One figure per parameter with 2x2 panels (one per metric).

    Each panel shows all environment curves overlapping on the SAME axes
    with distinct colors/markers/linestyles and a shared legend.
    One PNG file is saved per parameter.

    Metrics plotted:
        - Path Distance (m)
        - Max Curvature (1/m)
        - Min Clearance (m)
        - Computation Time (s)

    Args:
        all_param_results: dict  param_name -> {env -> {value -> [metrics]}}
        param_sweeps:      dict  param_name -> {"key": ..., "values": [...]}
        env_names:         list of environment names
        metrics_to_plot:   ignored (kept for API compatibility)
    """
    METRICS = [
        ("distance",         "Path Distance (m)"),
        ("max_curvature",    "Max Curvature (1/m)"),
        ("min_clearance",    "Min Clearance (m)"),
        ("computation_time", "Computation Time (s)"),
    ]

    # Color-blind-friendly palette + markers
    env_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    env_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '<', '>']
    env_linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

    for pname, sweep_info in param_sweeps.items():
        values = sweep_info["values"]
        env_results = all_param_results[pname]  # env -> val -> [metrics]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes_flat = axes.flatten()

        for mi, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes_flat[mi]

            for ei, env in enumerate(env_names):
                color = env_colors[ei % len(env_colors)]
                marker = env_markers[ei % len(env_markers)]
                ls = env_linestyles[ei % len(env_linestyles)]
                means, stds = [], []
                for v in values:
                    vals = [r[metric_key] for r in env_results[env][v]
                            if not np.isnan(r[metric_key])]
                    if vals:
                        means.append(np.mean(vals))
                        stds.append(np.std(vals))
                    else:
                        means.append(float('nan'))
                        stds.append(0.0)
                means = np.array(means)
                stds = np.array(stds)

                ax.plot(values, means, marker=marker, linestyle=ls,
                        color=color, linewidth=1.8, markersize=6,
                        markeredgecolor='k', markeredgewidth=0.4,
                        label=env)
                ax.fill_between(values, means - stds, means + stds,
                                alpha=0.12, color=color)

            ax.set_xlabel(pname, fontsize=10)
            ax.set_ylabel(metric_label, fontsize=10)
            #ax.set_title(metric_label, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(labelsize=9)

        # Shared legend outside the plot area (below all panels)
        handles, labels = axes_flat[0].get_legend_handles_labels()
        #fig.suptitle(f"Sensitivity: {pname}", fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
        fig.legend(handles, labels, loc='lower center',
                   ncol=min(len(env_names), 5), fontsize=9, frameon=True,
                   edgecolor='gray', fancybox=True,
                   bbox_to_anchor=(0.5, 0.0))
        safe_param = pname.replace(" ", "_").replace("/", "_").replace("*", "Star")
        fname = f"Sensitivity_{safe_param}.png"
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


def _sensitivity_worker(args):
    """Worker function for parallel sensitivity analysis.
    Runs ONE epoch of solve_patrol_nsga_all with given kwargs on a given environment.
    Each worker sets up its own global state (grid, dist_field) to avoid races.
    Returns metrics dict."""
    diff, nsga_kwargs, epoch_seed = args

    # Each worker gets its own copy of global state
    global grid, dist_field
    grid, waypoints = create_patrol_environment(diff)
    dist_field = ndi.distance_transform_edt(1 - grid)

    # Set unique seed per epoch for reproducibility without correlation
    random.seed(epoch_seed)
    np.random.seed(epoch_seed)

    t0 = time.time()
    p_nsga = solve_patrol_nsga_all(waypoints, **nsga_kwargs)
    elapsed = time.time() - t0

    m = _evaluate_nsga_path(p_nsga["Adaptive"])
    m["computation_time"] = elapsed
    return m


def run_sensitivity_mode(num_epochs=10):
    """Mode 3: Sensitivity analysis for A* initialized NSGA-II.

    For each hyper-parameter, sweep across a range of values while keeping
    all other parameters at their defaults.  At each setting, run
    `num_epochs` independent trials on every environment and collect:
      - Path distance (Adaptive winner)
      - Max curvature
      - Min obstacle clearance
      - Computation time

    Uses multiprocessing to parallelize across epochs for significant speedup
    on multi-core machines (e.g. Google Colab).

    Outputs: per-parameter line-plots (mean ± std) and a JSON summary.
    """
    global grid, dist_field

    n_workers = max(1, os.cpu_count() - 1)  # Leave 1 core for OS
    print(f"Detected {os.cpu_count()} CPU cores — using {n_workers} parallel workers")

    ENV_DIFFICULTIES = ["Easy", "Moderate-I",
                        "Moderate-II", "Moderate-III",
                        "Hard"]

    # ---- Define parameter sweep ranges ----
    PARAM_SWEEPS = {
        "Population Size":    {"key": "pop_size",    "values": [20, 40, 60, 80, 100]},
        "Generations":        {"key": "ngen",        "values": [10, 20, 40, 60, 80]},
        "Crossover Prob":     {"key": "cxpb",        "values": [0.5, 0.6, 0.7, 0.8, 0.9]},
        "Mutation Prob":      {"key": "mutpb",       "values": [0.1, 0.2, 0.3, 0.4, 0.5]},
        "Control Points":    {"key": "n_ctrl_pts",  "values": [3, 5, 7, 9, 11]},
        "A* Seed Ratio":     {"key": "seed_ratio",  "values": [0.0, 0.3, 0.5, 0.7, 1.0]},
        "Mutation Sigma":    {"key": "mut_sigma",   "values": [0.5, 1.0, 2.0, 3.0, 5.0]},
    }

    # Default values (must match run_nsga_ii defaults)
    DEFAULTS = {
        "pop_size": 60, "ngen": 40, "cxpb": 0.7, "mutpb": 0.3,
        "n_ctrl_pts": 5, "seed_ratio": 0.7, "mut_sigma": 2.0, "indpb": 0.2,
    }

    # Total job estimate
    total_jobs = sum(len(s["values"]) for s in PARAM_SWEEPS.values()) * len(ENV_DIFFICULTIES) * num_epochs
    est_serial_hours = total_jobs * 40 / 3600
    est_parallel_hours = est_serial_hours / n_workers
    print(f"Total jobs: {total_jobs}  |  Est. serial: {est_serial_hours:.1f}h  "
          f"|  Est. parallel ({n_workers} cores): {est_parallel_hours:.1f}h")

    master_results = {}  # param_name -> env -> value -> list[metrics]
    all_param_env_results = {}  # param_name -> {env -> {value -> [metrics]}}

    for param_name, sweep_info in PARAM_SWEEPS.items():
        key = sweep_info["key"]
        values = sweep_info["values"]
        print(f"\n{'='*70}")
        print(f"SENSITIVITY PARAMETER: {param_name} ({key})")
        print(f"  Sweep values: {values}")
        print(f"  Epochs per setting: {num_epochs}")
        print(f"{'='*70}")

        all_env_results = {}  # env -> value -> list[metrics]

        for diff in ENV_DIFFICULTIES:
            print(f"\n  Environment: {diff}")
            # Set global state for single-process fallback
            grid, waypoints = create_patrol_environment(diff)
            dist_field = ndi.distance_transform_edt(1 - grid)

            results_for_env = {}  # value -> list[metrics]

            # Build ALL tasks for this environment (across all param values & epochs)
            tasks = []  # list of (val, epoch, worker_args)
            for val in values:
                kwargs = dict(DEFAULTS)
                kwargs[key] = val
                for epoch in range(num_epochs):
                    epoch_seed = hash((param_name, diff, val, epoch)) % (2**31)
                    worker_args = (diff, kwargs, epoch_seed)
                    tasks.append((val, epoch, worker_args))

            # Execute in parallel
            task_results = {}  # (val, epoch) -> metrics
            if n_workers > 1:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    future_to_key = {}
                    for val, epoch, worker_args in tasks:
                        future = executor.submit(_sensitivity_worker, worker_args)
                        future_to_key[future] = (val, epoch)

                    completed = 0
                    for future in as_completed(future_to_key):
                        val, epoch = future_to_key[future]
                        try:
                            m = future.result()
                            task_results[(val, epoch)] = m
                            completed += 1
                            col_str = "" if m["collision_pts"] == 0 else f" COL={m['collision_pts']}"
                            print(f"    {key}={val}  epoch {epoch+1}/{num_epochs}  "
                                  f"dist={m['distance']:.2f}m  k={m['max_curvature']:.3f}  "
                                  f"clr={m['min_clearance']:.3f}m  t={m['computation_time']:.2f}s"
                                  f"{col_str}  [{completed}/{len(tasks)}]")
                        except Exception as e:
                            print(f"    {key}={val}  epoch {epoch+1}  FAILED: {e}")
                            task_results[(val, epoch)] = {
                                "distance": float('nan'), "max_curvature": float('nan'),
                                "min_clearance": float('nan'), "collision_pts": 0,
                                "computation_time": 0.0,
                            }
            else:
                # Single-core fallback (sequential)
                for val, epoch, worker_args in tasks:
                    m = _sensitivity_worker(worker_args)
                    task_results[(val, epoch)] = m
                    col_str = "" if m["collision_pts"] == 0 else f" COL={m['collision_pts']}"
                    print(f"    {key}={val}  epoch {epoch+1}/{num_epochs}  "
                          f"dist={m['distance']:.2f}m  k={m['max_curvature']:.3f}  "
                          f"clr={m['min_clearance']:.3f}m  t={m['computation_time']:.2f}s{col_str}")

            # Reassemble results by param value
            for val in values:
                epoch_metrics = [task_results[(val, ep)] for ep in range(num_epochs)
                                 if (val, ep) in task_results]
                results_for_env[val] = epoch_metrics

            all_env_results[diff] = results_for_env

            # Per-environment individual plots
            plot_sensitivity(results_for_env, param_name, values, diff,
                             metrics_to_plot=["distance", "max_curvature",
                                              "min_clearance", "computation_time"])

        # Combined multi-environment plots (no-op, deferred to single figure)
        plot_sensitivity_combined(all_env_results, param_name, values, ENV_DIFFICULTIES)

        # Store for the single unified plot at the end
        all_param_env_results[param_name] = all_env_results

        master_results[param_name] = {}
        for diff in ENV_DIFFICULTIES:
            master_results[param_name][diff] = {}
            for val in values:
                metrics_list = all_env_results[diff][val]
                master_results[param_name][diff][str(val)] = {
                    "distance_mean":   float(np.mean([m["distance"] for m in metrics_list])),
                    "distance_std":    float(np.std([m["distance"] for m in metrics_list])),
                    "max_curvature_mean": float(np.mean([m["max_curvature"] for m in metrics_list])),
                    "max_curvature_std":  float(np.std([m["max_curvature"] for m in metrics_list])),
                    "min_clearance_mean": float(np.mean([m["min_clearance"] for m in metrics_list])),
                    "min_clearance_std":  float(np.std([m["min_clearance"] for m in metrics_list])),
                    "time_mean":       float(np.mean([m["computation_time"] for m in metrics_list])),
                    "time_std":        float(np.std([m["computation_time"] for m in metrics_list])),
                    "collision_rate":  float(np.mean([1 if m["collision_pts"] > 0 else 0
                                                      for m in metrics_list])),
                }

    # Save complete results
    with open("sensitivity_results.json", "w") as f:
        json.dump(master_results, f, indent=4)
    print("\nSaved sensitivity_results.json")

    # --- Summary table ---
    print(f"\n{'='*70}")
    print("SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'='*70}")
    for param_name in PARAM_SWEEPS:
        print(f"\n  Parameter: {param_name}")
        values = PARAM_SWEEPS[param_name]["values"]
        print(f"    {'Value':>10s}  {'Dist(m)':>10s}  {'MaxK':>10s}  {'Clr(m)':>10s}  {'Time(s)':>10s}  {'ColRate':>8s}")
        for val in values:
            d_means, k_means, c_means, t_means, col_rates = [], [], [], [], []
            for diff in ENV_DIFFICULTIES:
                entry = master_results[param_name][diff][str(val)]
                d_means.append(entry["distance_mean"])
                k_means.append(entry["max_curvature_mean"])
                c_means.append(entry["min_clearance_mean"])
                t_means.append(entry["time_mean"])
                col_rates.append(entry["collision_rate"])
            print(f"    {val:>10}  {np.mean(d_means):10.2f}  {np.mean(k_means):10.3f}  "
                  f"{np.mean(c_means):10.3f}  {np.mean(t_means):10.2f}  "
                  f"{np.mean(col_rates):8.2%}")

    print("\nSensitivity analysis complete.")

    # ---- SINGLE UNIFIED FIGURE for all parameters ----
    plot_sensitivity_all(all_param_env_results, PARAM_SWEEPS, ENV_DIFFICULTIES)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    # Required on Windows for multiprocessing (spawn) to work correctly
    multiprocessing.freeze_support()

    # Usage:
    #   python patrollingAlgorithms.py single            -> 1 epoch, path figures
    #   python patrollingAlgorithms.py test               -> 100 epochs, boxplot & raincloud
    #   python patrollingAlgorithms.py sensitivity [N]    -> sensitivity analysis (N epochs, default 30)
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "single"

    if mode == "single":
        print("=== MODE: Single Epoch (path figures) ===")
        run_single_epoch_mode()
    elif mode == "test":
        print("=== MODE: 100-Epoch Test (boxplot & raincloud) ===")
        run_test_mode(num_epochs=100)
    elif mode == "sensitivity":
        n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        print(f"=== MODE: Sensitivity Analysis ({n_ep} epochs per setting) ===")
        run_sensitivity_mode(num_epochs=n_ep)
    else:
        print(f"Unknown mode '{mode}'. Use 'single', 'test', or 'sensitivity'.")
        sys.exit(1)