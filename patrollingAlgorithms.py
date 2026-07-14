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
import csv
import datetime
import platform
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

def _linear_resample_through_points(control_points, num_samples=100):
    """Piecewise-linear path through control points for controlled B-Spline ablation."""
    pts = np.array(control_points, dtype=float)
    if len(pts) < 2:
        return pts
    if len(pts) == 2:
        return np.linspace(pts[0], pts[1], num_samples)
    seg_lens = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cum_len = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = cum_len[-1]
    if total < 1e-9:
        return np.repeat(pts[:1], num_samples, axis=0)
    sample_s = np.linspace(0.0, total, num_samples)
    y = np.interp(sample_s, cum_len, pts[:, 0])
    x = np.interp(sample_s, cum_len, pts[:, 1])
    result = np.column_stack((y, x))
    result[:, 0] = np.clip(result[:, 0], 0, GRID_SIZE - 1)
    result[:, 1] = np.clip(result[:, 1], 0, GRID_SIZE - 1)
    result[0] = pts[0]
    result[-1] = pts[-1]
    return result


def _trajectory_from_control_points(control_points, num_samples=100, smoothing=0,
                                    spline_mode="bspline"):
    """Return a path using either the default B-Spline or a reduced linear variant."""
    mode = (spline_mode or "bspline").lower()
    if mode in {"linear", "reduced", "reduced_bspline", "no_bspline"}:
        return _linear_resample_through_points(control_points, num_samples)
    return common_spline(control_points, num_samples=num_samples, smoothing=smoothing)

def sparsify_path(path, min_dist=3.0):
    if len(path) < 2: return path
    new_path = [path[0]]
    for p in path[1:-1]:
        if np.linalg.norm(np.array(p) - np.array(new_path[-1])) > min_dist:
            new_path.append(p)
    new_path.append(path[-1])
    return np.array(new_path)

def curvature_limit_respline(path, num_samples=None):
    """Re-spline the path with increasing smoothing until curvature â‰¤ MAX_CURVATURE.
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
    Uses the discrete approximation: curvature â‰ˆ turning_angle / segment_length.
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
        # Already collision-free â€” apply curvature-limiting re-spline + kinematic smoothing
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

def construct_adaptive_path(champion_inds, obstacles_list, spline_mode="bspline", temperature=0.3):
    """Build adaptive path as a smooth weighted blend of champion paths.

    Instead of discrete point-by-point selection (which causes abrupt
    switches and sharp turns), this computes per-point quality scores for
    each champion, applies heavy temporal smoothing so the weights change
    gradually, and produces a soft-weighted blend of all champion paths.
    This inherently produces kinematically-feasible paths for non-holonomic
    two-wheeled robots because the output varies smoothly in space."""
    interpolated_paths = []
    for ind in champion_inds:
        pts = _trajectory_from_control_points(ind, PLOT_SAMPLES, spline_mode=spline_mode)
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

    # Heavy temporal smoothing â€” prevents rapid weight fluctuation
    kernel = max(min_len // 5, 80)
    for ci in range(n_champs):
        scores[ci] = ndi.uniform_filter1d(scores[ci].astype(float),
                                          size=kernel, mode='nearest')

    # Softmin weights (low temperature â†’ sharper selection of best champion)
    temperature = max(float(temperature), 1e-6)
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
        path = _trajectory_from_control_points(full_pts, CHECK_SAMPLES)
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
    K_ATT = 0.3    # Attractive gain (Î¾) â€” moderate pull toward goal
    K_REP = 1.0    # Repulsive gain (Î·) â€” obstacle push
    D0 = 4.0       # Repulsive influence distance (grid cells)
    APF_SCALE = 0.15  # Scale factor for APF force application

    start_arr = np.array(start, dtype=float)
    goal_arr = np.array(goal, dtype=float)

    def evaluate_whale(ctrl_pts):
        """Evaluate a whale (set of control points) as a full spline path."""
        sorted_pts = ctrl_pts[np.argsort(np.linalg.norm(ctrl_pts - start_arr, axis=1))]
        full_pts = np.vstack([start_arr, sorted_pts, goal_arr])
        path = _trajectory_from_control_points(full_pts, CHECK_SAMPLES)
        length = get_path_length(path)
        safety = check_safety_vectorized(path)
        smooth = get_smoothness_cost(path)
        curv = get_curvature_penalty(path) + get_curvature_integral_penalty(path)
        return 30.0 * length + 1000.0 * safety + 10.0 * smooth + curv

    def compute_apf_force(point):
        """Compute combined APF force: attractive + repulsive.
        Attractive: F_att = Î¾ Â· (q_goal - q) / |q_goal - q|
        Repulsive:  F_rep = Î· Â· (1/d - 1/d0) Â· (1/dÂ²) Â· nÌ‚   when d < d0
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
                A_vec = 2 * a_coeff * r1 - a_coeff   # Eq: A = 2aÂ·r - a
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
# Eq 15 (hybrid velocity): V = wÂ·V + c1Â·r1Â·(Whale* - X_woa) + c2Â·r2Â·(gbest - X_woa)
# Eq 20 (cost): f(Â·) = f_p Â· (1 + Î¼ Â· v_L)
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
        path = _trajectory_from_control_points(full_pts, CHECK_SAMPLES)
        f_p = get_path_length(path)
        v_L = calculate_violation_paper(path) # Uses SUM (tuned from paper's MEAN, Eq 18)
        
        # Eq 20: cost = f_p * (1 + Î¼ * v_L). Paper Î¼=100; tuned to 10000 for safety.
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
            A = 2*a*r1 - a; C = 2*r2  # Eq 9: A = 2aÂ·r-a, C = 2r
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
    spline_mode = kwargs.get('spline_mode', 'bspline')
    objective_mode = kwargs.get('objective_mode', 'five_objective')
    adaptive_mode = kwargs.get('adaptive_mode', 'softmin')
    postprocess_enabled = kwargs.get('postprocess_enabled', True)
    softmin_temperature = kwargs.get('softmin_temperature', 0.3)

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
        # Fallback: approach from startâ†’goal direction
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
        path = _trajectory_from_control_points(full_pts, CHECK_SAMPLES, spline_mode=spline_mode)
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
        
        if objective_mode == "length_only":
            return (length, length, length, length, length)
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
    
    if adaptive_mode == "fixed_length_champion":
        adaptive_coords = champions[0]
    elif adaptive_mode == "fixed_safety_champion":
        adaptive_coords = champions[4]
    else:
        adaptive_coords = construct_adaptive_path(
            champions, obstacles_list, spline_mode=spline_mode,
            temperature=softmin_temperature)

    # adaptive_coords is about 100 smooth control points in the full method; use
    # approximating spline only when the B-Spline representation is enabled.
    adaptive_smoothing = len(adaptive_coords) * 0.5 if (spline_mode or "").lower() == "bspline" else 0
    if len(adaptive_coords) > 2:
        path_adaptive = _trajectory_from_control_points(
            adaptive_coords, PLOT_SAMPLES, smoothing=adaptive_smoothing,
            spline_mode=spline_mode)
    else:
        path_adaptive = _trajectory_from_control_points(
            champions[0], PLOT_SAMPLES, spline_mode=spline_mode)

    if postprocess_enabled:
        path_adaptive = ensure_collision_free(path_adaptive)
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
        p = _trajectory_from_control_points(champ, PLOT_SAMPLES, spline_mode=spline_mode)
        if postprocess_enabled:
            p = ensure_collision_free(p)
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
        # Aisle width â‰ˆ 7-10 cells (0.7-1.0 m) â€” passable but demanding.

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
        # Doorway gap: rows 22-28 (7 cells â‰ˆ 0.7 m effective width).
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
        #   â€¢ Shelf aisles (WP2 between A-row shelves, WP8 between B/C shelves)
        #   â€¢ Central doorway bottleneck (WP4)
        #   â€¢ Near U-shape trap (WP3 sits just above the U opening)
        #   â€¢ Inside C-shape concavity (WP6 â€” must enter AND exit)
        #   â€¢ Close to pillars (WP2 near pillar@18,10; WP5 near pillar@18,40;
        #     WP8 near pillar@32,10)
        # The loop 1â†’2â†’3â†’4â†’5â†’6â†’7â†’8â†’1 requires â‰¥4 crossings of the
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

def solve_patrol_single(algo_func, waypoints, timing_records=None, timing_context=None):
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
        segment_t0 = time.perf_counter()
        segment, end_dir = algo_func(start, end, start_dir=curr_dir, goal_dir=goal_dir)
        segment_elapsed = time.perf_counter() - segment_t0
        if timing_records is not None:
            _append_segment_timing(timing_records, timing_context, i, start, end,
                                   segment_elapsed)
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

def solve_patrol_nsga_all(waypoints, timing_records=None, timing_context=None, **nsga_kwargs):
    full_paths = {k: [] for k in ["Length", "Smooth", "Effort", "Centered", "Safe", "Adaptive"]}
    targets = waypoints.copy()
    targets.append(waypoints[0])
    curr_dir = FIXED_START_DIR.copy()
    postprocess_enabled = nsga_kwargs.get("postprocess_enabled", True)
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
        segment_t0 = time.perf_counter()
        seg_dict, end_dir = run_nsga_ii(start, end, start_dir=curr_dir, goal_dir=goal_dir, **nsga_kwargs)
        segment_elapsed = time.perf_counter() - segment_t0
        if timing_records is not None:
            _append_segment_timing(
                timing_records, timing_context, i, start, end, segment_elapsed,
                generated_outputs=[
                    "NSGA-II Length", "NSGA-II Smooth", "NSGA-II Effort",
                    "NSGA-II Centered", "NSGA-II Safe", "NSGA-II Adaptive"
                ]
            )
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

    if postprocess_enabled:
        # Post-concatenation junction smoothing for the Adaptive path
        # This eliminates sharp turns where segments meet at waypoints
        for k in ["Adaptive"]:
            path = full_paths[k]
            if len(path) < 10:
                continue
            for jidx in junction_indices[k]:
                # Smooth a local window around each junction
                window = min(80, len(path) // 8)  # about 80 points each side
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
    
    ax.set_xticklabels(algo_names, rotation=35, ha='right', fontsize=14)
    ax.set_ylabel('Path Distance (m)', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    #ax.set_title(f'Path Distance Distribution â€” {diff} (100 epochs, no fixed seed)', fontsize=13)
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
    ax.set_xticklabels(algo_names, rotation=35, ha='right', fontsize=14)
    ax.set_ylabel('Path Distance (m)', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    #ax.set_title(f'Raincloud â€” Path Distance Distribution â€” {diff} (100 epochs, no fixed seed)', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(f"Raincloud_{diff}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved Raincloud_{diff}.png")


def plot_metric_boxplots(all_metrics, diff, algo_names, algo_colors):
    """Box plots for each metric (curvature, clearance, smoothness, effort, centering)."""
    PLOTS = [
        ("max_curvature", "Max Curvature (1/m)", MAX_CURVATURE),
        ("min_clearance", "Min Obstacle Clearance (m)", MIN_SAFE_DIST),
        ("smoothness", "Smoothness Cost (rad)", None),
        ("wheel_effort", "Wheel Effort", None),
        ("centering", "Centering Score", None),
    ]
    for metric_key, ylabel, threshold in PLOTS:
        fig, ax = plt.subplots(figsize=(14, 7))
        data = [all_metrics[metric_key][name] for name in algo_names]

        bp = ax.boxplot(data, patch_artist=True, notch=True, widths=0.6,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=4, alpha=0.5))

        for patch, color in zip(bp['boxes'], [algo_colors[n] for n in algo_names]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        if threshold is not None:
            ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
                       label=f'Threshold = {threshold:.2f}')
            ax.legend(fontsize=12)

        ax.set_xticklabels(algo_names, rotation=35, ha='right', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        fig.tight_layout()
        fname = f"Boxplot_{metric_key}_{diff}.png"
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_radar_chart(all_metrics, diff, algo_names, algo_colors):
    """Radar/spider chart comparing all algorithms across normalized metrics."""
    RADAR_METRICS = ["distance", "max_curvature", "smoothness", "wheel_effort", "centering"]
    RADAR_LABELS = ["Path Length", "Max Curvature", "Smoothness", "Wheel Effort", "Centering"]

    # Compute mean per algorithm per metric
    means = {}
    for name in algo_names:
        means[name] = [float(np.mean(all_metrics[mk][name])) for mk in RADAR_METRICS]

    # Normalize: for each metric, divide by the max across algorithms (lower = better)
    n_metrics = len(RADAR_METRICS)
    max_vals = []
    for j in range(n_metrics):
        mv = max(means[name][j] for name in algo_names)
        max_vals.append(mv if mv > 1e-12 else 1.0)
    norm_means = {}
    for name in algo_names:
        norm_means[name] = [means[name][j] / max_vals[j] for j in range(n_metrics)]

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    for name in algo_names:
        values = norm_means[name] + norm_means[name][:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=name, color=algo_colors[name])
        ax.fill(angles, values, alpha=0.08, color=algo_colors[name])

    ax.set_thetagrids(np.degrees(angles[:-1]), RADAR_LABELS, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    fig.tight_layout()
    fname = f"Radar_{diff}.png"
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_normalized_bar_chart(all_metrics, diff, algo_names, algo_colors):
    """Grouped bar chart with normalized metric values (best=1.0 per metric)."""
    METRICS = ["distance", "max_curvature", "min_clearance", "smoothness", "wheel_effort"]
    LABELS = ["Path Length", "Max Curvature", "Min Clearance\n(higher=better)", "Smoothness", "Wheel Effort"]

    means = {}
    for name in algo_names:
        means[name] = [float(np.mean(all_metrics[mk][name])) for mk in METRICS]

    # Normalize: each metric scaled so best algorithm = 1.0
    n_m = len(METRICS)
    best_vals = []
    for j in range(n_m):
        if METRICS[j] == "min_clearance":
            # higher is better -> best = max
            best_vals.append(max(means[name][j] for name in algo_names))
        else:
            # lower is better -> best = min
            best_vals.append(min(means[name][j] for name in algo_names))

    norm = {}
    for name in algo_names:
        row = []
        for j in range(n_m):
            bv = best_vals[j] if best_vals[j] > 1e-12 else 1.0
            if METRICS[j] == "min_clearance":
                row.append(means[name][j] / bv)  # higher = 1.0
            else:
                row.append(bv / means[name][j] if means[name][j] > 1e-12 else 0.0)  # lower = 1.0
            row[-1] = min(row[-1], 1.0)
        norm[name] = row

    x = np.arange(n_m)
    width = 0.08
    n_algos = len(algo_names)
    offsets = np.linspace(-(n_algos - 1) / 2 * width, (n_algos - 1) / 2 * width, n_algos)

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, name in enumerate(algo_names):
        ax.bar(x + offsets[i], norm[name], width, label=name,
               color=algo_colors[name], edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=13)
    ax.set_ylabel('Normalized Score (1.0 = best)', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    ax.legend(fontsize=9, ncol=3, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()
    fname = f"NormalizedBar_{diff}.png"
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------
# MODE 1: SINGLE EPOCH â€” path figures (like original)
# ---------------------------------------------------------
def run_single_epoch_mode():
    """Run 1 epoch per environment with a fixed seed, produce path overlay figures."""
    seedno = 50
    random.seed(seedno); np.random.seed(seedno)
    global grid, dist_field
    ENV_DIFFICULTIES = ["Easy", "Moderate-I", "Moderate-III", "Moderate-II", "Hard"]
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


def compute_all_metrics(path):
    """Compute all comparison metrics for a given path.
    Returns dict with distance, max_curvature, min_clearance, smoothness, wheel_effort."""
    obstacles_list = np.argwhere(grid == 1)
    iy = np.clip(np.round(path[:, 0]).astype(int), 0, GRID_SIZE - 1)
    ix = np.clip(np.round(path[:, 1]).astype(int), 0, GRID_SIZE - 1)
    dists_m = dist_field[iy, ix] * GRID_RES
    return {
        "distance": get_path_length(path),
        "max_curvature": get_max_curvature(path),
        "min_clearance": float(np.min(dists_m)),
        "smoothness": get_smoothness_cost(path),
        "wheel_effort": calculate_wheel_effort(path),
        "centering": calculate_centering_score(path, obstacles_list),
    }


# ---------------------------------------------------------
# MODE 2: 100-EPOCH TEST â€” boxplot & raincloud
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
    result["Standard GA"] = {**compute_all_metrics(p_ga), "time": time.time() - t0}

    t0 = time.time()
    p_ab = solve_patrol_single(run_ab_woa, waypoints)
    result["AB-WOA-APF"] = {**compute_all_metrics(p_ab), "time": time.time() - t0}

    t0 = time.time()
    p_wp = solve_patrol_single(run_woa_pso, waypoints)
    result["HWPSO"] = {**compute_all_metrics(p_wp), "time": time.time() - t0}

    # --- Run NSGA-II ---
    t0 = time.time()
    p_nsga = solve_patrol_nsga_all(waypoints)
    t_nsga = time.time() - t0

    for k in ["Length", "Smooth", "Effort", "Centered", "Safe", "Adaptive"]:
        result[f"NSGA-II {k}"] = {**compute_all_metrics(p_nsga[k]), "time": t_nsga}

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

    n_workers = max(1, os.cpu_count() - 2)
    print(f"Detected {os.cpu_count()} CPU cores â€” using {n_workers} parallel workers")

    ENV_DIFFICULTIES = ["Easy", "Moderate-I", "Moderate-III", "Moderate-II", "Hard"]

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
        print(f"Environment: {diff}  â€”  {num_epochs} epochs (no fixed seed)")
        print(f"{'='*60}")

        # Set global state for single-process fallback
        grid, waypoints = create_patrol_environment(diff)
        dist_field = ndi.distance_transform_edt(1 - grid)

        # Storage for all metrics across epochs
        METRIC_KEYS = ["distance", "max_curvature", "min_clearance", "smoothness", "wheel_effort", "centering"]
        all_metrics = {mk: {name: [] for name in ALGO_NAMES} for mk in METRIC_KEYS}
        all_distances = {name: [] for name in ALGO_NAMES}  # kept for backward compat
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
                for mk in METRIC_KEYS:
                    all_metrics[mk][name].append(result[name][mk])

        # --- Summary statistics for all metrics ---
        METRIC_LABELS = {
            "distance": ("Distance", "m"),
            "max_curvature": ("Max Curvature", "1/m"),
            "min_clearance": ("Min Clearance", "m"),
            "smoothness": ("Smoothness", "rad"),
            "wheel_effort": ("Wheel Effort", "-"),
            "centering": ("Centering", "-"),
        }
        full_summary = {}
        for mk in METRIC_KEYS:
            label, unit = METRIC_LABELS[mk]
            print(f"\n  === {label} Summary for {diff} ({unit}) ===")
            print(f"  {'Algorithm':25s}  {'Mean':>10s}  {'Std':>10s}  {'Min':>10s}  {'Max':>10s}  {'Median':>10s}")
            mk_summary = {}
            for name in ALGO_NAMES:
                arr = np.array(all_metrics[mk][name])
                mk_summary[name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "median": float(np.median(arr)),
                    "all": [float(v) for v in arr],
                }
                print(f"  {name:25s}  {np.mean(arr):10.4f}  {np.std(arr):10.4f}  "
                      f"{np.min(arr):10.4f}  {np.max(arr):10.4f}  {np.median(arr):10.4f}")
            full_summary[mk] = mk_summary

        # Save raw data as JSON (all metrics)
        with open(f"all_metrics_{diff}.json", "w") as f:
            json.dump(full_summary, f, indent=4)
        print(f"  Saved all_metrics_{diff}.json")

        # Save distances separately for backward compatibility
        with open(f"distances_{diff}.json", "w") as f:
            json.dump(full_summary["distance"], f, indent=4)
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
        plot_metric_boxplots(all_metrics, diff, ALGO_NAMES, ALGO_COLORS)
        plot_radar_chart(all_metrics, diff, ALGO_NAMES, ALGO_COLORS)
        plot_normalized_bar_chart(all_metrics, diff, ALGO_NAMES, ALGO_COLORS)

    print("\nTest mode complete.")


# ---------------------------------------------------------
# MODE 3: SENSITIVITY ANALYSIS for A* Initialized NSGA-II
# ---------------------------------------------------------
def _evaluate_nsga_path(path):
    """Compute sensitivity metrics for one NSGA-II Adaptive patrol path."""
    metrics = compute_all_metrics(path)
    iy = np.clip(np.round(path[:, 0]).astype(int), 0, GRID_SIZE - 1)
    ix = np.clip(np.round(path[:, 1]).astype(int), 0, GRID_SIZE - 1)
    collision = int(np.sum(grid[iy, ix] == 1))
    metrics["collision_count"] = collision
    metrics["collision_pts"] = collision  # Backward-compatible alias used by existing plots/logs.
    return metrics


def plot_sensitivity(results, param_name, param_values, env_name, metrics_to_plot=None):
    """No-op â€” all plotting handled by plot_sensitivity_all."""
    pass


def plot_sensitivity_combined(all_env_results, param_name, param_values,
                              env_names, metrics_to_plot=None):
    """No-op â€” all plotting handled by plot_sensitivity_all."""
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
    m["seed"] = int(epoch_seed)
    return m


SENSITIVITY_BASE_SEED = 20260703
SENSITIVITY_RHO_TAU_PARAMS = {"A* Seed Ratio", "Softmin Temperature"}
SENSITIVITY_METRIC_FIELDS = [
    ("distance", "path_length"),
    ("max_curvature", "maximum_curvature"),
    ("min_clearance", "minimum_clearance"),
    ("smoothness", "smoothness"),
    ("wheel_effort", "wheel_effort"),
    ("centering", "centering"),
    ("computation_time", "computation_time_seconds"),
]


def _sensitivity_env_configs():
    return [
        {"index": 0, "manuscript_environment": "Level-1", "internal_environment": "Easy"},
        {"index": 1, "manuscript_environment": "Level-2", "internal_environment": "Moderate-I"},
        {"index": 2, "manuscript_environment": "Level-3", "internal_environment": "Moderate-III"},
        {"index": 3, "manuscript_environment": "Level-4", "internal_environment": "Moderate-II"},
        {"index": 4, "manuscript_environment": "Level-5", "internal_environment": "Hard"},
    ]


def _select_sensitivity_sweeps(param_sweeps, scope):
    normalized = (scope or "all").lower()
    if normalized in {"all", "full"}:
        return param_sweeps
    if normalized in {"rho_tau", "rhotau"}:
        return {k: v for k, v in param_sweeps.items() if k in SENSITIVITY_RHO_TAU_PARAMS}
    if normalized in {"rho", "seed", "seed_ratio"}:
        return {"A* Seed Ratio": param_sweeps["A* Seed Ratio"]}
    if normalized in {"tau", "temperature", "softmin_temperature"}:
        return {"Softmin Temperature": param_sweeps["Softmin Temperature"]}
    valid = "all, rho_tau, rho, tau"
    raise ValueError(f"Unknown sensitivity scope '{scope}'. Valid values: {valid}")


def _sensitivity_seed(param_index, env_index, epoch_index):
    return int(SENSITIVITY_BASE_SEED + param_index * 100000 + env_index * 10000 + epoch_index + 1)


def _finite_metric_values(metrics_list, key):
    values = []
    for metrics in metrics_list:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isnan(value):
            values.append(value)
    return values


def _stats_dict(values):
    if not values:
        return {"mean": "", "std": "", "min": "", "max": "", "median": ""}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def _summarize_sensitivity_metrics(metrics_list):
    summary = {
        "episode_count": len(metrics_list),
        "valid_count": len(_finite_metric_values(metrics_list, "distance")),
    }
    for source_key, output_key in SENSITIVITY_METRIC_FIELDS:
        stats = _stats_dict(_finite_metric_values(metrics_list, source_key))
        for stat_name, value in stats.items():
            summary[f"{output_key}_{stat_name}"] = value
    collisions = _finite_metric_values(metrics_list, "collision_count")
    summary["collision_count_sum"] = int(np.sum(collisions)) if collisions else 0
    summary["collision_count_mean"] = float(np.mean(collisions)) if collisions else 0.0
    summary["collision_rate"] = float(np.mean([1 if v > 0 else 0 for v in collisions])) if collisions else 0.0
    summary["distance_mean"] = summary["path_length_mean"]
    summary["distance_std"] = summary["path_length_std"]
    summary["max_curvature_mean"] = summary["maximum_curvature_mean"]
    summary["max_curvature_std"] = summary["maximum_curvature_std"]
    summary["min_clearance_mean"] = summary["minimum_clearance_mean"]
    summary["min_clearance_std"] = summary["minimum_clearance_std"]
    summary["time_mean"] = summary["computation_time_seconds_mean"]
    summary["time_std"] = summary["computation_time_seconds_std"]
    seeds = [int(m["seed"]) for m in metrics_list if m.get("seed") is not None]
    summary["seed_min"] = min(seeds) if seeds else ""
    summary["seed_max"] = max(seeds) if seeds else ""
    return summary


def _sensitivity_summary_rows(master_results, param_sweeps, env_configs, param_names, num_epochs):
    rows = []
    cfg_by_internal = {cfg["internal_environment"]: cfg for cfg in env_configs}
    for param_name in param_names:
        if param_name not in master_results:
            continue
        sweep = param_sweeps[param_name]
        parameter_code = "rho" if param_name == "A* Seed Ratio" else "tau"
        for internal_env in master_results[param_name]:
            cfg = cfg_by_internal.get(internal_env, {"manuscript_environment": internal_env,
                                                     "internal_environment": internal_env})
            for value in sweep["values"]:
                entry = master_results[param_name][internal_env].get(str(value))
                if not entry:
                    continue
                row = {
                    "parameter": parameter_code,
                    "parameter_label": param_name,
                    "parameter_key": sweep["key"],
                    "parameter_value": value,
                    "manuscript_environment": cfg["manuscript_environment"],
                    "internal_environment": cfg["internal_environment"],
                    "requested_epochs": int(num_epochs),
                }
                row.update(entry)
                rows.append(row)
    return rows


def _sensitivity_summary_fieldnames():
    fields = ["parameter", "parameter_label", "parameter_key", "parameter_value",
              "manuscript_environment", "internal_environment", "requested_epochs",
              "episode_count", "valid_count", "seed_min", "seed_max"]
    for _, output_key in SENSITIVITY_METRIC_FIELDS:
        for stat in ["mean", "std", "min", "max", "median"]:
            fields.append(f"{output_key}_{stat}")
    fields.extend(["collision_count_sum", "collision_count_mean", "collision_rate"])
    return fields


def _latex_num(value, digits=3):
    if value == "" or value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "--"


def _write_sensitivity_latex(rho_rows, tau_rows):
    bs = chr(92)
    lines = [
        bs + "begin{table}[!htbp]",
        bs + "caption{Summary of rho and softmin-temperature sensitivity results. Values are means over completed fixed-seed episodes for each environment and parameter setting.}",
        bs + "label{tab:rho_tau_sensitivity}",
        bs + "centering",
        bs + "scriptsize",
        bs + "begin{tabular}{lllrrrrr}",
        bs + "toprule",
        "Parameter & Environment & Value & Length & Max. curv. & Min. clear. & Coll. & Time (s) " + bs + bs,
        bs + "midrule",
    ]
    for row in rho_rows + tau_rows:
        lines.append(
            f"{_latex_escape(row['parameter'])} & {_latex_escape(row['manuscript_environment'])} & "
            f"{_latex_num(row['parameter_value'], 2)} & "
            f"{_latex_num(row.get('path_length_mean'))} & "
            f"{_latex_num(row.get('maximum_curvature_mean'))} & "
            f"{_latex_num(row.get('minimum_clearance_mean'))} & "
            f"{int(row.get('collision_count_sum') or 0)} & "
            f"{_latex_num(row.get('computation_time_seconds_mean'))} " + bs + bs)
    lines.extend([bs + "bottomrule", bs + "end{tabular}", bs + "end{table}"])
    with open("sensitivity_rho_tau_for_latex.tex", "w", encoding="utf-8") as f:
        f.write(chr(10).join(lines) + chr(10))
    return "sensitivity_rho_tau_for_latex.tex"


def _write_sensitivity_report(metadata, rho_rows, tau_rows, generated_files):
    report = f"""# Rho/Tau Sensitivity Report

Generated: {metadata['generated_at']}

## Analysis Scope

- Fixed-parameter sensitivity and robustness evidence for the A* seed ratio and softmin temperature.
- Fixed seeds, reported parameter grids, and saved summary outputs support reproducibility.

## Command

```text
{metadata['command_used']}
```

## Parameter Sweeps

- rho / A* seed ratio: {metadata['rho_values']}
- tau / softmin temperature: {metadata['tau_values']}

Default values are preserved and included: rho = 0.7 and tau = 0.3.

## Metrics Recorded

- path length
- maximum curvature
- minimum clearance
- smoothness
- wheel effort
- centering
- collision count and collision rate
- computation time

## Seed Policy

Base seed: {metadata['base_seed']}. For each parameter, environment, and episode, the seed is computed deterministically as base_seed + parameter_index * 100000 + environment_index * 10000 + episode_index + 1, where parameter_index is taken from the full sensitivity parameter list so seeds remain stable when running only rho/tau. The same episode seed is reused across values of the same parameter to isolate parameter effects from random-seed effects.

## Environment Coverage

The sweep covers all five manuscript environments, mapped internally as Level-1/Easy, Level-2/Moderate-I, Level-3/Moderate-III, Level-4/Moderate-II, and Level-5/Hard.

## Important Limitation

These outputs quantify fixed-parameter sensitivity only. They do not implement adaptive dynamic parameter tuning, and no optimal range should be claimed unless the completed numerical outputs support it.

## Generated Files

{chr(10).join('- ' + filename for filename in generated_files)}

## Completed Row Counts

- rho summary rows: {len(rho_rows)}
- tau summary rows: {len(tau_rows)}
"""
    with open("sensitivity_rho_tau_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    return "sensitivity_rho_tau_report.md"


def _write_rho_tau_sensitivity_outputs(master_results, param_sweeps, env_configs, num_epochs, scope, command_used):
    rho_rows = _sensitivity_summary_rows(master_results, param_sweeps, env_configs,
                                         ["A* Seed Ratio"], num_epochs)
    tau_rows = _sensitivity_summary_rows(master_results, param_sweeps, env_configs,
                                         ["Softmin Temperature"], num_epochs)
    fieldnames = _sensitivity_summary_fieldnames()
    generated = []
    _write_csv_rows("sensitivity_rho_summary.csv", rho_rows, fieldnames)
    generated.append("sensitivity_rho_summary.csv")
    _write_csv_rows("sensitivity_tau_summary.csv", tau_rows, fieldnames)
    generated.append("sensitivity_tau_summary.csv")
    generated.append(_write_sensitivity_latex(rho_rows, tau_rows))
    metadata = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "command_used": command_used,
        "scope": scope,
        "num_epochs": int(num_epochs),
        "base_seed": SENSITIVITY_BASE_SEED,
        "rho_values": param_sweeps.get("A* Seed Ratio", {}).get("values", []),
        "tau_values": param_sweeps.get("Softmin Temperature", {}).get("values", []),
    }
    report_file = "sensitivity_rho_tau_report.md"
    generated.append(_write_sensitivity_report(metadata, rho_rows, tau_rows,
                                               generated + [report_file]))
    return generated

def run_sensitivity_mode(num_epochs=10, scope="all"):
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

    Outputs: per-parameter line-plots (mean Â± std) and a JSON summary.
    """
    global grid, dist_field

    n_workers = max(1, os.cpu_count() - 2)  # Leave 2 core for OS
    print(f"Detected {os.cpu_count()} CPU cores â€” using {n_workers} parallel workers")

    ENV_CONFIGS = _sensitivity_env_configs()
    ENV_DIFFICULTIES = [cfg["internal_environment"] for cfg in ENV_CONFIGS]

    # ---- Define parameter sweep ranges ----
    ALL_PARAM_SWEEPS = {
        "Population Size":      {"key": "pop_size",            "values": [20, 40, 60, 80, 100]},
        "Generations":          {"key": "ngen",                "values": [10, 20, 40, 60, 80]},
        "Crossover Prob":       {"key": "cxpb",                "values": [0.5, 0.6, 0.7, 0.8, 0.9]},
        "Mutation Prob":        {"key": "mutpb",               "values": [0.1, 0.2, 0.3, 0.4, 0.5]},
        "Control Points":       {"key": "n_ctrl_pts",          "values": [3, 5, 7, 9, 11]},
        "A* Seed Ratio":        {"key": "seed_ratio",          "values": [0.0, 0.25, 0.5, 0.7, 0.9, 1.0]},
        "Softmin Temperature":  {"key": "softmin_temperature", "values": [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]},
        "Mutation Sigma":       {"key": "mut_sigma",           "values": [0.5, 1.0, 2.0, 3.0, 5.0]},
    }
    PARAM_SWEEPS = _select_sensitivity_sweeps(ALL_PARAM_SWEEPS, scope)
    PARAM_INDEX = {name: idx for idx, name in enumerate(ALL_PARAM_SWEEPS)}

    # Default values (must match run_nsga_ii defaults)
    DEFAULTS = {
        "pop_size": 60, "ngen": 40, "cxpb": 0.7, "mutpb": 0.3,
        "n_ctrl_pts": 5, "seed_ratio": 0.7, "mut_sigma": 2.0, "indpb": 0.2,
        "softmin_temperature": 0.3,
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
        param_index = PARAM_INDEX[param_name]
        key = sweep_info["key"]
        values = sweep_info["values"]
        print(f"\n{'='*70}")
        print(f"SENSITIVITY PARAMETER: {param_name} ({key})")
        print(f"  Sweep values: {values}")
        print(f"  Epochs per setting: {num_epochs}")
        print(f"{'='*70}")

        all_env_results = {}  # env -> value -> list[metrics]

        for env_index, diff in enumerate(ENV_DIFFICULTIES):
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
                    epoch_seed = _sensitivity_seed(param_index, env_index, epoch)
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
                                "min_clearance": float('nan'), "smoothness": float('nan'),
                                "wheel_effort": float('nan'), "centering": float('nan'),
                                "collision_count": 0, "collision_pts": 0,
                                "computation_time": float('nan'),
                                "seed": _sensitivity_seed(param_index, env_index, epoch),
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
        for env_index, diff in enumerate(ENV_DIFFICULTIES):
            master_results[param_name][diff] = {}
            for val in values:
                metrics_list = all_env_results[diff][val]
                master_results[param_name][diff][str(val)] = _summarize_sensitivity_metrics(metrics_list)

    # Save complete results
    with open("sensitivity_results.json", "w") as f:
        json.dump(master_results, f, indent=4)
    print("\nSaved sensitivity_results.json")
    generated_sensitivity_files = _write_rho_tau_sensitivity_outputs(
        master_results=master_results,
        param_sweeps=PARAM_SWEEPS,
        env_configs=ENV_CONFIGS,
        num_epochs=num_epochs,
        scope=scope,
        command_used=" ".join(sys.argv),
    )
    print("Saved rho/tau sensitivity outputs:")
    for filename in generated_sensitivity_files:
        print(f"  {filename}")

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
            for env_index, diff in enumerate(ENV_DIFFICULTIES):
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
# MODE 4: RESUMABLE SEGMENT TIMING INSTRUMENTATION
def _segment_env_configs():
    return [
        {"index": 0, "manuscript_environment": "Level-1", "internal_environment": "Easy"},
        {"index": 1, "manuscript_environment": "Level-2", "internal_environment": "Moderate-I"},
        {"index": 2, "manuscript_environment": "Level-3", "internal_environment": "Moderate-III"},
        {"index": 3, "manuscript_environment": "Level-4", "internal_environment": "Moderate-II"},
        {"index": 4, "manuscript_environment": "Level-5", "internal_environment": "Hard"},
    ]


def _select_segment_envs(selected_env):
    configs = _segment_env_configs()
    if selected_env is None or selected_env.lower() == "all":
        return configs
    selected = selected_env.lower()
    matches = [cfg for cfg in configs
               if selected in {cfg["internal_environment"].lower(),
                               cfg["manuscript_environment"].lower()}]
    if not matches:
        valid = ", ".join([cfg["internal_environment"] for cfg in configs] +
                          [cfg["manuscript_environment"] for cfg in configs])
        raise ValueError(f"Unknown segment timing environment '{selected_env}'. Valid values: all, {valid}")
    return matches


def _waypoint_to_json(point):
    values = np.asarray(point).tolist()
    result = []
    for value in values:
        value = float(value)
        result.append(int(value) if value.is_integer() else value)
    return result


def _safe_env_filename(name):
    return name.replace(" ", "_")


def _set_timing_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _stats_from_times(times):
    arr = np.array(times, dtype=float)
    return {
        "count": int(len(arr)),
        "mean_seconds": float(np.mean(arr)) if len(arr) else 0.0,
        "std_seconds": float(np.std(arr)) if len(arr) else 0.0,
        "min_seconds": float(np.min(arr)) if len(arr) else 0.0,
        "max_seconds": float(np.max(arr)) if len(arr) else 0.0,
        "median_seconds": float(np.median(arr)) if len(arr) else 0.0,
        "sum_seconds": float(np.sum(arr)) if len(arr) else 0.0,
    }


def _route_summary_from_segments(segment_records, route_runtime_seconds,
                                 generated_outputs=None):
    if not segment_records:
        return {}
    stats = _stats_from_times([r["segment_planning_time_seconds"] for r in segment_records])
    first = segment_records[0]
    outputs = generated_outputs or first.get("generated_outputs", [])
    return {
        "manuscript_environment": first["manuscript_environment"],
        "internal_environment": first["internal_environment"],
        "source_environment": first["internal_environment"],
        "environment_name": first["manuscript_environment"],
        "method": first["method"],
        "episode_id": first["episode_id"],
        "run_id": first["run_id"],
        "episode_index": first["episode_index"],
        "seed": first["seed"],
        "route_runtime_seconds": float(route_runtime_seconds),
        "segment_time_sum_seconds": stats["sum_seconds"],
        "segment_time_mean_seconds": stats["mean_seconds"],
        "segment_time_std_seconds": stats["std_seconds"],
        "segment_time_min_seconds": stats["min_seconds"],
        "segment_time_max_seconds": stats["max_seconds"],
        "segment_time_median_seconds": stats["median_seconds"],
        "segment_count": stats["count"],
        "route_minus_segment_sum_seconds": float(route_runtime_seconds - stats["sum_seconds"]),
        "generated_outputs": list(outputs),
        "nsga_multi_output_call": bool(outputs),
        "runtime_scope": "full_route_with_segment_breakdown",
    }


def _summarize_segment_records(segment_records):
    groups = {}
    for record in segment_records:
        key = (record["manuscript_environment"], record["internal_environment"],
               record["method"], tuple(record.get("generated_outputs", [])))
        groups.setdefault(key, []).append(record)
    rows = []
    for key in sorted(groups):
        records = groups[key]
        stats = _stats_from_times([r["segment_planning_time_seconds"] for r in records])
        episode_ids = sorted(set(r["episode_id"] for r in records))
        seeds = sorted(set(int(r["seed"]) for r in records if r.get("seed") is not None))
        per_episode = []
        for episode_id in episode_ids:
            per_episode.append(len([r for r in records if r["episode_id"] == episode_id]))
        rows.append({
            "manuscript_environment": key[0],
            "internal_environment": key[1],
            "source_environment": key[1],
            "method": key[2],
            "generated_outputs": list(key[3]),
            "episode_count": len(episode_ids),
            "run_count": len(episode_ids),
            "segments_per_episode": int(per_episode[0]) if per_episode else 0,
            "segment_sample_count": stats["count"],
            "mean_segment_time_seconds": stats["mean_seconds"],
            "std_segment_time_seconds": stats["std_seconds"],
            "min_segment_time_seconds": stats["min_seconds"],
            "max_segment_time_seconds": stats["max_seconds"],
            "median_segment_time_seconds": stats["median_seconds"],
            "sum_segment_time_seconds": stats["sum_seconds"],
            "seed_min": min(seeds) if seeds else "",
            "seed_max": max(seeds) if seeds else "",
            "timing_scope": "single_segment_planning_call",
        })
    return rows


def _flatten_csv_value(value):
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return value


def _write_csv_rows(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _flatten_csv_value(row.get(name, ""))
                             for name in fieldnames})


def _latex_escape(text):
    return str(text).replace("&", r"\&").replace("_", r"\_")


def _summary_lookup(rows):
    return {(row["manuscript_environment"], row["method"]): row for row in rows}

# ---------------------------------------------------------
def _segment_method_selection(method_filter):
    method_filter = (method_filter or "all").lower()
    competitors = [("Standard GA", run_ga_standard),
                   ("AB-WOA-APF", run_ab_woa),
                   ("HWPSO", run_woa_pso)]
    if method_filter == "proposed_only":
        return [], True, ["NSGA-II multi-output"], method_filter
    if method_filter == "competitors_only":
        return competitors, False, [name for name, _ in competitors], method_filter
    if method_filter == "all":
        return competitors, True, [name for name, _ in competitors] + ["NSGA-II multi-output"], method_filter
    raise ValueError("Unknown segment timing method filter "
                     f"'{method_filter}'. Valid values: all, proposed_only, competitors_only")


def _segment_checkpoint_filename(manuscript_env):
    return f"calculation_times_segment_{_safe_env_filename(manuscript_env)}.json"


def _episode_index_from_text(text):
    if text is None:
        return None
    for token in reversed(str(text).replace("-", "_").split("_")):
        if token.isdigit():
            return int(token)
    return None


def _normalize_segment_record(record, cfg):
    result = dict(record)
    result["manuscript_environment"] = result.get("manuscript_environment") or cfg["manuscript_environment"]
    result["internal_environment"] = result.get("internal_environment") or result.get("source_environment") or cfg["internal_environment"]
    result["source_environment"] = result["internal_environment"]
    result["environment_name"] = result["manuscript_environment"]
    episode_index = result.get("episode_index") or _episode_index_from_text(result.get("episode_id")) or _episode_index_from_text(result.get("run_id"))
    result["episode_index"] = int(episode_index) if episode_index is not None else None
    if not result.get("episode_id") and result["episode_index"] is not None:
        result["episode_id"] = f"{result['manuscript_environment']}_episode_{result['episode_index']:03d}"
    result["run_id"] = result.get("run_id") or result.get("episode_id", "")
    result["segment_index"] = int(result.get("segment_index", -1))
    if result.get("seed") is not None:
        result["seed"] = int(result["seed"])
    result["generated_outputs"] = list(result.get("generated_outputs", []))
    result["nsga_multi_output_call"] = bool(result["generated_outputs"])
    return result


def _normalize_route_record(record, cfg):
    result = dict(record)
    result["manuscript_environment"] = result.get("manuscript_environment") or cfg["manuscript_environment"]
    result["internal_environment"] = result.get("internal_environment") or result.get("source_environment") or cfg["internal_environment"]
    result["source_environment"] = result["internal_environment"]
    result["environment_name"] = result["manuscript_environment"]
    episode_index = result.get("episode_index") or _episode_index_from_text(result.get("episode_id")) or _episode_index_from_text(result.get("run_id"))
    result["episode_index"] = int(episode_index) if episode_index is not None else None
    if not result.get("episode_id") and result["episode_index"] is not None:
        result["episode_id"] = f"{result['manuscript_environment']}_episode_{result['episode_index']:03d}"
    result["run_id"] = result.get("run_id") or result.get("episode_id", "")
    if result.get("seed") is not None:
        result["seed"] = int(result["seed"])
    result["generated_outputs"] = list(result.get("generated_outputs", []))
    result["nsga_multi_output_call"] = bool(result["generated_outputs"])
    return result


def _segment_record_key(record):
    return (record.get("manuscript_environment"), record.get("internal_environment"),
            record.get("method"), int(record.get("episode_index") or -1),
            int(record.get("seed") if record.get("seed") is not None else -1),
            int(record.get("segment_index") or -1))


def _route_record_key(record):
    return (record.get("manuscript_environment"), record.get("internal_environment"),
            record.get("method"), int(record.get("episode_index") or -1),
            int(record.get("seed") if record.get("seed") is not None else -1))


def _dedupe_records(records, key_func):
    deduped = {}
    for record in records:
        deduped[key_func(record)] = record
    return [deduped[key] for key in sorted(deduped)]


def _load_segment_env_checkpoint(cfg):
    filename = _segment_checkpoint_filename(cfg["manuscript_environment"])
    if not os.path.exists(filename):
        return [], [], False
    with open(filename, "r", encoding="utf-8") as f:
        payload = json.load(f)
    segments = [_normalize_segment_record(r, cfg) for r in payload.get("segment_records", [])]
    routes = [_normalize_route_record(r, cfg) for r in payload.get("route_summaries", [])]
    return _dedupe_records(segments, _segment_record_key), _dedupe_records(routes, _route_record_key), True


def _filter_records_by_methods(records, method_names):
    allowed = set(method_names)
    return [record for record in records if record.get("method") in allowed]


def _completed_episode_count(route_summaries, method_names):
    counts = []
    for method in method_names:
        episodes = {int(r["episode_index"]) for r in route_summaries
                    if r.get("method") == method and r.get("episode_index") is not None}
        counts.append(len(episodes))
    return min(counts) if counts else 0


def _segment_method_episode_seed(record, manuscript_env, internal_env, method, episode_index, seed):
    return (_route_record_key(record) ==
            (manuscript_env, internal_env, method, int(episode_index), int(seed)))


def _estimate_total_segment_calls(configs, num_runs, method_names):
    total = 0
    for cfg in configs:
        _, waypoints = create_patrol_environment(cfg["internal_environment"])
        total += len(waypoints) * int(num_runs) * len(method_names)
    return total


def _count_completed_segment_calls(configs, method_names, num_runs):
    total = 0
    for cfg in configs:
        segments, _, _ = _load_segment_env_checkpoint(cfg)
        for record in segments:
            if (record.get("method") in method_names and
                    int(record.get("episode_index") or 0) <= int(num_runs)):
                total += 1
    return total


def _make_progress_callback(progress_state):
    def _callback(record):
        progress_state["completed_segment_calls"] += 1
        completed = progress_state["completed_segment_calls"]
        total = progress_state["total_segment_calls"]
        elapsed_all = time.perf_counter() - progress_state["started_perf"]
        eta = "not_available"
        if completed > 0 and total > 0:
            eta = f"{(elapsed_all / completed) * max(total - completed, 0):.1f}s"
        print("    progress | "
              f"env={record['manuscript_environment']} ({record['internal_environment']}) | "
              f"method={record['method']} | episode={record['episode_index']}/{progress_state['num_runs']} | "
              f"seed={record['seed']} | segment={record['segment_index']}/{progress_state['current_waypoint_count']} | "
              f"elapsed={record['segment_planning_time_seconds']:.3f}s | completed={completed}/{total} | eta={eta}")
    return _callback


def _append_segment_timing(records, timing_context, segment_index, start, goal,
                           elapsed_seconds, generated_outputs=None):
    context = timing_context or {}
    waypoint_count = context.get("waypoint_count")
    from_wp = segment_index + 1
    to_wp = 1 if waypoint_count and from_wp == waypoint_count else from_wp + 1
    outputs = generated_outputs or context.get("generated_outputs", [])
    manuscript_env = context.get("manuscript_environment", "")
    internal_env = context.get("internal_environment", context.get("source_environment", ""))
    record = {
        "manuscript_environment": manuscript_env,
        "internal_environment": internal_env,
        "source_environment": internal_env,
        "environment_name": manuscript_env,
        "method": context.get("method", "unknown"),
        "episode_id": context.get("episode_id", context.get("run_id", "")),
        "run_id": context.get("run_id", context.get("episode_id", "")),
        "episode_index": context.get("episode_index"),
        "seed": context.get("seed"),
        "segment_index": int(segment_index + 1),
        "segment_label": f"WP_{from_wp}->WP_{to_wp}",
        "start_waypoint": _waypoint_to_json(start),
        "goal_waypoint": _waypoint_to_json(goal),
        "segment_planning_time_seconds": float(elapsed_seconds),
        "generated_outputs": list(outputs),
        "nsga_multi_output_call": bool(outputs),
    }
    records.append(record)
    callback = context.get("progress_callback")
    if callback is not None:
        callback(record)

def _load_existing_full_route_runtime_lookup():
    lookup = {}
    path = "runtime_summary_by_environment.csv"
    if not os.path.exists(path):
        return lookup
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            env = row.get("manuscript_environment", "")
            method = row.get("method", "")
            if env and method:
                lookup[(env, method)] = {
                    "mean_seconds": row.get("runtime_mean_seconds", ""),
                    "std_seconds": row.get("runtime_std_seconds", ""),
                    "source_file": row.get("source_file", path),
                    "matched_method": method,
                }
    return lookup


def _existing_full_route_for_method(lookup, manuscript_env, method):
    candidates = [method]
    if method == "NSGA-II multi-output":
        candidates = ["NSGA-II Adaptive", "NSGA-II Length"]
    for candidate in candidates:
        if (manuscript_env, candidate) in lookup:
            return lookup[(manuscript_env, candidate)]
    return {"mean_seconds": "", "std_seconds": "", "source_file": "", "matched_method": ""}


def _summarize_route_records(route_summaries):
    existing_lookup = _load_existing_full_route_runtime_lookup()
    groups = {}
    for record in route_summaries:
        key = (record["manuscript_environment"], record["internal_environment"],
               record["method"], tuple(record.get("generated_outputs", [])))
        groups.setdefault(key, []).append(record)
    rows = []
    for key in sorted(groups):
        records = groups[key]
        route_stats = _stats_from_times([r["route_runtime_seconds"] for r in records])
        sum_stats = _stats_from_times([r["segment_time_sum_seconds"] for r in records])
        mean_stats = _stats_from_times([r["segment_time_mean_seconds"] for r in records])
        max_stats = _stats_from_times([r["segment_time_max_seconds"] for r in records])
        overhead_stats = _stats_from_times([r["route_minus_segment_sum_seconds"] for r in records])
        seeds = sorted(set(int(r["seed"]) for r in records if r.get("seed") is not None))
        segment_counts = sorted(set(int(r["segment_count"]) for r in records))
        existing = _existing_full_route_for_method(existing_lookup, key[0], key[2])
        rows.append({
            "manuscript_environment": key[0],
            "internal_environment": key[1],
            "source_environment": key[1],
            "method": key[2],
            "generated_outputs": list(key[3]),
            "episode_count": len(records),
            "run_count": len(records),
            "segments_per_episode": segment_counts[0] if len(segment_counts) == 1 else ";".join(str(v) for v in segment_counts),
            "route_runtime_mean_seconds": route_stats["mean_seconds"],
            "route_runtime_std_seconds": route_stats["std_seconds"],
            "route_runtime_min_seconds": route_stats["min_seconds"],
            "route_runtime_max_seconds": route_stats["max_seconds"],
            "route_runtime_median_seconds": route_stats["median_seconds"],
            "summed_segment_runtime_mean_seconds": sum_stats["mean_seconds"],
            "summed_segment_runtime_std_seconds": sum_stats["std_seconds"],
            "summed_segment_runtime_min_seconds": sum_stats["min_seconds"],
            "summed_segment_runtime_max_seconds": sum_stats["max_seconds"],
            "summed_segment_runtime_median_seconds": sum_stats["median_seconds"],
            "mean_segment_runtime_mean_seconds": mean_stats["mean_seconds"],
            "mean_segment_runtime_std_seconds": mean_stats["std_seconds"],
            "mean_segment_runtime_min_seconds": mean_stats["min_seconds"],
            "mean_segment_runtime_max_seconds": mean_stats["max_seconds"],
            "mean_segment_runtime_median_seconds": mean_stats["median_seconds"],
            "maximum_segment_runtime_mean_seconds": max_stats["mean_seconds"],
            "maximum_segment_runtime_std_seconds": max_stats["std_seconds"],
            "maximum_segment_runtime_min_seconds": max_stats["min_seconds"],
            "maximum_segment_runtime_max_seconds": max_stats["max_seconds"],
            "maximum_segment_runtime_median_seconds": max_stats["median_seconds"],
            "route_minus_segment_sum_mean_seconds": overhead_stats["mean_seconds"],
            "route_minus_segment_sum_std_seconds": overhead_stats["std_seconds"],
            "route_minus_segment_sum_min_seconds": overhead_stats["min_seconds"],
            "route_minus_segment_sum_max_seconds": overhead_stats["max_seconds"],
            "route_minus_segment_sum_median_seconds": overhead_stats["median_seconds"],
            "existing_full_route_method": existing["matched_method"],
            "existing_full_route_runtime_mean_seconds": existing["mean_seconds"],
            "existing_full_route_runtime_std_seconds": existing["std_seconds"],
            "existing_full_route_runtime_source": existing["source_file"],
            "seed_min": min(seeds) if seeds else "",
            "seed_max": max(seeds) if seeds else "",
            "runtime_scope": "full_route_with_segment_breakdown_summary",
        })
    return rows


def _write_single_segment_latex(summary_rows, route_summary_rows,
                                path="single_segment_time_for_latex.tex"):
    route_lookup = _summary_lookup(route_summary_rows)
    lines = [
        r"\begin{table}[!htbp]",
        r"\caption{Runtime and true single-segment planning time from fixed-seed checkpointed episodes. The NSGA-II row represents one multi-output NSGA-II optimization call per segment that jointly generates Length, Smooth, Effort, Centered, Safe, and Adaptive paths.}",
        r"\label{tab:single_segment_time}",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"\textbf{Environment} & \textbf{Method} & \textbf{Episodes} & \textbf{Segments} & \textbf{Segment time (s)} & \textbf{Full-route time (s)} \\",
        r"\hline",
    ]
    for row in summary_rows:
        route_row = route_lookup.get((row["manuscript_environment"], row["method"]), {})
        segment_mean_std = f"{row['mean_segment_time_seconds']:.2f}$\\pm${row['std_segment_time_seconds']:.2f}"
        existing_mean = route_row.get("existing_full_route_runtime_mean_seconds", "")
        existing_std = route_row.get("existing_full_route_runtime_std_seconds", "")
        if existing_mean != "" and existing_std != "":
            route_mean_std = f"{float(existing_mean):.2f}$\\pm${float(existing_std):.2f}"
        else:
            route_mean = route_row.get("route_runtime_mean_seconds", 0.0)
            route_std = route_row.get("route_runtime_std_seconds", 0.0)
            route_mean_std = f"{float(route_mean):.2f}$\\pm${float(route_std):.2f}"
        lines.append(
            f"{_latex_escape(row['manuscript_environment'])} & "
            f"{_latex_escape(row['method'])} & "
            f"{row['episode_count']} & {row['segments_per_episode']} & "
            f"{segment_mean_std} & {route_mean_std} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_segment_timing_aggregate_outputs(segment_records, route_summaries):
    segment_summary_rows = _summarize_segment_records(segment_records)
    route_summary_rows = _summarize_route_records(route_summaries)
    segment_fields = [
        "manuscript_environment", "internal_environment", "source_environment", "method",
        "generated_outputs", "episode_count", "run_count", "segments_per_episode",
        "segment_sample_count", "mean_segment_time_seconds", "std_segment_time_seconds",
        "min_segment_time_seconds", "max_segment_time_seconds", "median_segment_time_seconds",
        "sum_segment_time_seconds", "seed_min", "seed_max", "timing_scope",
    ]
    route_fields = [
        "manuscript_environment", "internal_environment", "source_environment", "method",
        "generated_outputs", "episode_count", "run_count", "segments_per_episode",
        "route_runtime_mean_seconds", "route_runtime_std_seconds", "route_runtime_min_seconds",
        "route_runtime_max_seconds", "route_runtime_median_seconds",
        "summed_segment_runtime_mean_seconds", "summed_segment_runtime_std_seconds",
        "summed_segment_runtime_min_seconds", "summed_segment_runtime_max_seconds",
        "summed_segment_runtime_median_seconds", "mean_segment_runtime_mean_seconds",
        "mean_segment_runtime_std_seconds", "mean_segment_runtime_min_seconds",
        "mean_segment_runtime_max_seconds", "mean_segment_runtime_median_seconds",
        "maximum_segment_runtime_mean_seconds", "maximum_segment_runtime_std_seconds",
        "maximum_segment_runtime_min_seconds", "maximum_segment_runtime_max_seconds",
        "maximum_segment_runtime_median_seconds", "route_minus_segment_sum_mean_seconds",
        "route_minus_segment_sum_std_seconds", "route_minus_segment_sum_min_seconds",
        "route_minus_segment_sum_max_seconds", "route_minus_segment_sum_median_seconds",
        "existing_full_route_method", "existing_full_route_runtime_mean_seconds",
        "existing_full_route_runtime_std_seconds", "existing_full_route_runtime_source",
        "seed_min", "seed_max", "runtime_scope",
    ]
    _write_csv_rows("single_segment_time_summary.csv", segment_summary_rows, segment_fields)
    _write_csv_rows("route_vs_segment_runtime_summary.csv", route_summary_rows, route_fields)
    _write_single_segment_latex(segment_summary_rows, route_summary_rows)
    return segment_summary_rows, route_summary_rows

def _format_seconds(value):
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "not_available"


def _write_segment_timing_report(metadata, generated_files, route_summary_rows,
                                 path="segment_timing_instrumentation_report.md"):
    env_lines = "\n".join(f"- {cfg['manuscript_environment']} ({cfg['internal_environment']})"
                           for cfg in metadata["environment_configs"])
    method_lines = "\n".join(f"- {name}" for name in metadata["method_names"])
    manuscript_files = [name for name in generated_files if "Level-" in name]
    manuscript_file_lines = "\n".join(f"- `{name}`" for name in manuscript_files) or "- No Level checkpoint files found yet."
    generated_file_lines = "\n".join(f"- `{name}`" for name in generated_files)
    overhead_values = [row["route_minus_segment_sum_mean_seconds"] for row in route_summary_rows]
    overhead_max_values = [row["route_minus_segment_sum_max_seconds"] for row in route_summary_rows]
    overhead_mean = float(np.mean(overhead_values)) if overhead_values else 0.0
    overhead_max = float(np.max(overhead_max_values)) if overhead_max_values else 0.0
    report = f"""# Segment Timing Instrumentation Report

Date: {metadata['generated_at']}

Analysis scope:

- Running time, computational complexity, single-segment planning time, and online/real-time suitability.
- Computational overhead of segmented serial optimization and real-time limitations.

## Long-Run Policy

The 100-episode timing experiment is designed as a resumable local run. The script supports checkpointing so a long run can be interrupted and continued safely.

## Command Syntax

Recommended local commands for final proposed-method segment timing:

```text
python -u patrollingAlgorithms.py segment_timing 100 Level-1 proposed_only
python -u patrollingAlgorithms.py segment_timing 100 Level-2 proposed_only
python -u patrollingAlgorithms.py segment_timing 100 Level-3 proposed_only
python -u patrollingAlgorithms.py segment_timing 100 Level-4 proposed_only
python -u patrollingAlgorithms.py segment_timing 100 Level-5 proposed_only
```

To regenerate summaries from existing checkpoints without running planners:

```text
python -u patrollingAlgorithms.py segment_timing_aggregate proposed_only
```

Last command represented in this report: `{metadata['command_used']}`

## Checkpoint And Resume

- Results are written to `calculation_times_segment_Level-*.json` after every completed environment-method block.
- Resume keys use manuscript environment, internal environment, method, episode index, seed, and segment index.
- On resume, completed route records are skipped and duplicate segment/route records are deduplicated.
- If a method block is incomplete, its partial records are discarded before rerunning that same method/episode.

## Where Timing Was Inserted

- `solve_patrol_single(...)`: a timer wraps each per-segment call to the supplied planner function (`run_ga_standard`, `run_ab_woa`, or `run_woa_pso`).
- `solve_patrol_nsga_all(...)`: a timer wraps each per-segment call to `run_nsga_ii(...)`.
- `run_segment_timing_mode(...)`: route-level timers wrap the full call to `solve_patrol_single(...)` or `solve_patrol_nsga_all(...)`, so route runtime and the sum/mean/max of segment times are measured from the same instrumented run.

## Algorithmic Logic

No algorithmic logic, objective function, default parameter, waypoint definition, obstacle layout, environment geometry, post-processing operation, or optimizer setting was changed. The edit only standardizes manuscript-facing labels, records optional timing data, and adds checkpoint/resume controls.

## Timing Configuration

- Method filter: `{metadata['method_filter']}`
- Target episodes per selected environment/method: {metadata['num_runs']}
- Base seed: {metadata['base_seed']}
- Seed policy: each environment/episode/method receives a deterministic logged seed computed as `base_seed + environment_index * 10000 + episode_index * 100 + method_index`.
- Python: {metadata['python_version']}
- Platform: {metadata['platform']}
- Processor: {metadata['processor']}
- CPU count: {metadata['cpu_count']}

## Environment-Name Mapping

Internal environment identifiers are kept only for `create_patrol_environment(...)` and traceability. Manuscript-facing outputs use the following labels:

{env_lines}

## Method List

{method_lines}

The archived single-segment timing prioritizes `proposed_only`, because the existing 100-episode full-route timing outputs already provide method-level full-route runtime comparisons. `competitors_only` and `all` remain available for additional segment timing.

For NSGA-II, one segment timing record corresponds to the single `run_nsga_ii(...)` call that jointly produces Length, Smooth, Effort, Centered, Safe, and Adaptive outputs. These outputs are not counted as independent optimization runs.

## Runtime Analysis Versus Sensitivity Analysis

Runtime and single-segment timing are computational-cost analyses, not parameter-sensitivity experiments. The final local timing run should use 100 fixed-seed episodes to match the manuscript's main statistical comparison protocol. The separate sensitivity analysis may remain at 30 independent runs per parameter setting because it answers a different question about parameter robustness.

## Route Runtime Versus Segment Runtime

Route runtime measured during segment timing is saved together with the summed, mean, and maximum segment runtimes. When available, `route_vs_segment_runtime_summary.csv` also includes the existing full-route runtime mean and standard deviation from `runtime_summary_by_environment.csv`.

Across the currently aggregated route summaries, the mean route-minus-segment overhead is {_format_seconds(overhead_mean)} s and the maximum recorded aggregate overhead is {_format_seconds(overhead_max)} s.

## Online And Real-Time Limitations

These timing measurements do not by themselves establish embedded real-time suitability. The segmented serial optimization structure requires repeated iterative optimization calls over waypoint-to-waypoint segments; therefore, the measured runtime should be interpreted conservatively when discussing online use, low-power embedded deployment, or strict real-time navigation.

## Manuscript-Facing Files To Use

{manuscript_file_lines}

## All Generated Files

{generated_file_lines}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)


def _segment_env_payload(manuscript_env, internal_env, num_runs, base_seed,
                         env_segment_records, env_route_summaries, status,
                         method_filter, method_names):
    selected_routes = _filter_records_by_methods(env_route_summaries, method_names)
    selected_segments = _filter_records_by_methods(env_segment_records, method_names)
    completed = _completed_episode_count(selected_routes, method_names)
    target = max(int(num_runs), completed)
    return {
        "metadata": {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "mode": "segment_timing",
            "status": status,
            "method_filter": method_filter,
            "manuscript_environment": manuscript_env,
            "internal_environment": internal_env,
            "source_environment": internal_env,
            "num_runs_requested": int(num_runs),
            "num_episodes_requested": int(num_runs),
            "target_episode_count": target,
            "num_runs_completed": completed,
            "num_episodes_completed": completed,
            "base_seed": int(base_seed),
            "python_version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "environment_name_mapping": f"{internal_env} -> {manuscript_env}",
            "checkpoint_key": "manuscript_environment, internal_environment, method, episode_index, seed, segment_index",
            "note": "One NSGA-II segment call jointly produces Length, Smooth, Effort, Centered, Safe, and Adaptive outputs.",
        },
        "route_summaries": _dedupe_records(env_route_summaries, _route_record_key),
        "segment_records": _dedupe_records(env_segment_records, _segment_record_key),
        "single_segment_summary": _summarize_segment_records(selected_segments),
        "route_vs_segment_summary": _summarize_route_records(selected_routes),
    }


def _write_segment_env_json(manuscript_env, internal_env, num_runs, base_seed,
                            env_segment_records, env_route_summaries, status,
                            method_filter, method_names):
    filename = _segment_checkpoint_filename(manuscript_env)
    payload = _segment_env_payload(manuscript_env, internal_env, num_runs, base_seed,
                                   env_segment_records, env_route_summaries, status,
                                   method_filter, method_names)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    return filename


def _aggregate_available_segment_timing_outputs(configs=None, method_filter="all",
                                                num_runs=100, command_used="aggregate"):
    configs = configs or _segment_env_configs()
    _, _, method_names, normalized_filter = _segment_method_selection(method_filter)
    all_segments = []
    all_routes = []
    generated_files = []
    for cfg in configs:
        segments, routes, exists = _load_segment_env_checkpoint(cfg)
        if exists:
            selected_routes = _filter_records_by_methods(routes, method_names)
            completed = _completed_episode_count(selected_routes, method_names)
            status = "complete" if completed >= int(num_runs) else "partial"
            filename = _write_segment_env_json(
                cfg["manuscript_environment"], cfg["internal_environment"],
                num_runs, 20260701, segments, routes, status,
                normalized_filter, method_names)
            generated_files.append(filename)
            all_segments.extend(_filter_records_by_methods(segments, method_names))
            all_routes.extend(selected_routes)
    _, route_summary_rows = _write_segment_timing_aggregate_outputs(all_segments, all_routes)
    generated_files.extend(["single_segment_time_summary.csv",
                            "route_vs_segment_runtime_summary.csv",
                            "single_segment_time_for_latex.tex"])
    metadata = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "command_used": command_used,
        "environment_configs": configs,
        "num_runs": int(num_runs),
        "base_seed": 20260701,
        "method_filter": normalized_filter,
        "method_names": method_names,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor() or "not reported",
        "cpu_count": os.cpu_count() or "not reported",
    }
    generated_files.append("segment_timing_instrumentation_report.md")
    _write_segment_timing_report(metadata, generated_files, route_summary_rows)
    return generated_files


def run_segment_timing_aggregate_mode(method_filter="proposed_only", selected_env="all"):
    configs = _select_segment_envs(selected_env)
    generated_files = _aggregate_available_segment_timing_outputs(
        configs=configs,
        method_filter=method_filter,
        command_used=f"python -u patrollingAlgorithms.py segment_timing_aggregate {method_filter} {selected_env}",
    )
    print("Segment timing aggregation complete. Generated files:")
    for filename in generated_files:
        print(f"  {filename}")

def run_segment_timing_mode(num_runs=1, selected_env="all", method_filter="all"):
    """Measure route and true per-segment planning time without changing planners."""
    global grid, dist_field

    base_seed = 20260701
    configs = _select_segment_envs(selected_env)
    method_specs, include_nsga, method_names, normalized_filter = _segment_method_selection(method_filter)
    nsga_outputs = ["NSGA-II Length", "NSGA-II Smooth", "NSGA-II Effort",
                    "NSGA-II Centered", "NSGA-II Safe", "NSGA-II Adaptive"]
    total_calls = _estimate_total_segment_calls(configs, num_runs, method_names)
    done_calls = _count_completed_segment_calls(configs, method_names, num_runs)
    progress_state = {"started_perf": time.perf_counter(),
                      "completed_segment_calls": done_calls,
                      "total_segment_calls": total_calls,
                      "num_runs": int(num_runs),
                      "current_waypoint_count": 0}
    progress_callback = _make_progress_callback(progress_state)

    for cfg in configs:
        internal_env = cfg["internal_environment"]
        manuscript_env = cfg["manuscript_environment"]
        env_segments, env_routes, had_checkpoint = _load_segment_env_checkpoint(cfg)
        completed_routes = {_route_record_key(record) for record in env_routes}
        print(f"\n=== Segment timing: {manuscript_env} ({internal_env}), method_filter={normalized_filter} ===")
        if had_checkpoint:
            completed = _completed_episode_count(_filter_records_by_methods(env_routes, method_names), method_names)
            print(f"  Loaded checkpoint {_segment_checkpoint_filename(manuscript_env)} "
                  f"with {completed}/{num_runs} completed selected episodes.")

        for run_idx in range(int(num_runs)):
            episode_index = run_idx + 1
            episode_id = f"{manuscript_env}_episode_{episode_index:03d}"
            grid, waypoints = create_patrol_environment(internal_env)
            dist_field = ndi.distance_transform_edt(1 - grid)
            waypoint_count = len(waypoints)
            progress_state["current_waypoint_count"] = waypoint_count

            for method_index, (method_name, algo_func) in enumerate(method_specs):
                seed = base_seed + cfg["index"] * 10000 + run_idx * 100 + method_index
                route_key = (manuscript_env, internal_env, method_name, episode_index, seed)
                if route_key in completed_routes:
                    print(f"  skip | {episode_id} | {method_name} | seed={seed} already complete")
                    continue
                env_segments = [r for r in env_segments
                                if not _segment_method_episode_seed(r, manuscript_env, internal_env,
                                                                    method_name, episode_index, seed)]
                _set_timing_seed(seed)
                segment_records = []
                context = {"manuscript_environment": manuscript_env,
                           "internal_environment": internal_env,
                           "source_environment": internal_env,
                           "method": method_name,
                           "episode_id": episode_id,
                           "run_id": episode_id,
                           "episode_index": episode_index,
                           "seed": seed,
                           "waypoint_count": waypoint_count,
                           "progress_callback": progress_callback}
                route_t0 = time.perf_counter()
                solve_patrol_single(algo_func, waypoints,
                                    timing_records=segment_records,
                                    timing_context=context)
                route_elapsed = time.perf_counter() - route_t0
                route_summary = _route_summary_from_segments(segment_records, route_elapsed)
                env_segments.extend(segment_records)
                env_routes.append(route_summary)
                env_segments = _dedupe_records(env_segments, _segment_record_key)
                env_routes = _dedupe_records(env_routes, _route_record_key)
                completed_routes = {_route_record_key(record) for record in env_routes}
                _write_segment_env_json(manuscript_env, internal_env, num_runs, base_seed,
                                        env_segments, env_routes, "partial",
                                        normalized_filter, method_names)
                print(f"  saved | {episode_id} | {method_name}: route={route_elapsed:.3f}s, "
                      f"segments={route_summary['segment_time_sum_seconds']:.3f}s")

            if include_nsga:
                method_name = "NSGA-II multi-output"
                nsga_method_index = 3
                seed = base_seed + cfg["index"] * 10000 + run_idx * 100 + nsga_method_index
                route_key = (manuscript_env, internal_env, method_name, episode_index, seed)
                if route_key in completed_routes:
                    print(f"  skip | {episode_id} | {method_name} | seed={seed} already complete")
                    continue
                env_segments = [r for r in env_segments
                                if not _segment_method_episode_seed(r, manuscript_env, internal_env,
                                                                    method_name, episode_index, seed)]
                _set_timing_seed(seed)
                segment_records = []
                context = {"manuscript_environment": manuscript_env,
                           "internal_environment": internal_env,
                           "source_environment": internal_env,
                           "method": method_name,
                           "episode_id": episode_id,
                           "run_id": episode_id,
                           "episode_index": episode_index,
                           "seed": seed,
                           "waypoint_count": waypoint_count,
                           "generated_outputs": nsga_outputs,
                           "progress_callback": progress_callback}
                route_t0 = time.perf_counter()
                solve_patrol_nsga_all(waypoints, timing_records=segment_records,
                                      timing_context=context)
                route_elapsed = time.perf_counter() - route_t0
                route_summary = _route_summary_from_segments(segment_records, route_elapsed,
                                                             generated_outputs=nsga_outputs)
                env_segments.extend(segment_records)
                env_routes.append(route_summary)
                env_segments = _dedupe_records(env_segments, _segment_record_key)
                env_routes = _dedupe_records(env_routes, _route_record_key)
                completed_routes = {_route_record_key(record) for record in env_routes}
                _write_segment_env_json(manuscript_env, internal_env, num_runs, base_seed,
                                        env_segments, env_routes, "partial",
                                        normalized_filter, method_names)
                print(f"  saved | {episode_id} | {method_name}: route={route_elapsed:.3f}s, "
                      f"segments={route_summary['segment_time_sum_seconds']:.3f}s")

        selected_routes = _filter_records_by_methods(env_routes, method_names)
        completed = _completed_episode_count(selected_routes, method_names)
        status = "complete" if completed >= int(num_runs) else "partial"
        filename = _write_segment_env_json(manuscript_env, internal_env, num_runs, base_seed,
                                           env_segments, env_routes, status,
                                           normalized_filter, method_names)
        print(f"  Saved {filename} ({status}, selected completed episodes={completed}/{num_runs})")

    generated_files = _aggregate_available_segment_timing_outputs(
        configs=_segment_env_configs(),
        method_filter=normalized_filter,
        num_runs=num_runs,
        command_used=f"python -u patrollingAlgorithms.py segment_timing {int(num_runs)} {selected_env} {normalized_filter}")
    print("\nSegment timing mode complete. Generated files:")
    for filename in generated_files:
        print(f"  {filename}")


# ---------------------------------------------------------
# MODE 5: ABLATION EXPERIMENT INFRASTRUCTURE
# ---------------------------------------------------------
ABLATION_BASE_SEED = 20260702


def _ablation_variant_specs():
    return [
        {"key": "full_proposed", "name": "Full proposed method", "kwargs": {},
         "code_level_meaning": "Default proposed framework with all modules enabled."},
        {"key": "no_astar_initialization", "name": "No A* initialization",
         "kwargs": {"seed_ratio": 0.0},
         "code_level_meaning": "Sets seed_ratio=0.0 to remove A*-guided NSGA-II initialization."},
        {"key": "reduced_bspline_representation", "name": "Reduced B-Spline representation",
         "kwargs": {"spline_mode": "linear"},
         "code_level_meaning": "Uses piecewise-linear trajectory construction inside NSGA-II evaluation and output construction; safety repair may still invoke spline-based repair if needed."},
        {"key": "reduced_objective_nsga", "name": "Reduced-objective NSGA-II",
         "kwargs": {"objective_mode": "length_only"},
         "code_level_meaning": "Replaces the five distinct NSGA-II objectives with a length-only objective tuple."},
        {"key": "no_softmin_adaptive_fusion", "name": "No softmin adaptive fusion",
         "kwargs": {"adaptive_mode": "fixed_length_champion"},
         "code_level_meaning": "Disables softmin blending and uses the Length champion as a fixed final path."},
        {"key": "no_post_processing", "name": "No post-processing",
         "kwargs": {"postprocess_enabled": False},
         "code_level_meaning": "Disables collision repair, smoothing, obstacle escape, curvature re-splining, and junction smoothing after optimization."},
    ]


def _count_path_collisions(path):
    if path is None or len(path) == 0:
        return 0
    iy = np.clip(np.round(path[:, 0]).astype(int), 0, GRID_SIZE - 1)
    ix = np.clip(np.round(path[:, 1]).astype(int), 0, GRID_SIZE - 1)
    return int(np.sum(grid[iy, ix] == 1))


def _compute_ablation_metrics(path):
    metrics = compute_all_metrics(path)
    collisions = _count_path_collisions(path)
    return {
        "path_length": float(metrics["distance"]),
        "maximum_curvature": float(metrics["max_curvature"]),
        "minimum_obstacle_clearance": float(metrics["min_clearance"]),
        "smoothness": float(metrics["smoothness"]),
        "wheel_effort": float(metrics["wheel_effort"]),
        "centering": float(metrics["centering"]),
        "collision_count": collisions,
        "collision_flag": bool(collisions > 0),
    }


def _stats_for_field(records, field):
    values = [r.get(field) for r in records if r.get(field) is not None]
    if not values:
        return {"mean": "", "std": "", "min": "", "max": "", "median": ""}
    arr = np.array(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "min": float(np.min(arr)), "max": float(np.max(arr)),
            "median": float(np.median(arr))}


def _summarize_ablation_records(records):
    variant_lookup = {v["key"]: v for v in _ablation_variant_specs()}
    groups = {}
    for record in records:
        key = (record["manuscript_environment"], record["internal_environment"], record["variant_key"])
        groups.setdefault(key, []).append(record)
    rows = []
    metric_fields = ["path_length", "maximum_curvature", "minimum_obstacle_clearance",
                     "smoothness", "wheel_effort", "centering", "route_runtime_seconds"]
    for key in sorted(groups):
        env, internal_env, variant_key = key
        group = groups[key]
        variant = variant_lookup.get(variant_key, {"name": variant_key})
        success_records = [r for r in group if r.get("status") == "ok"]
        seeds = sorted(set(int(r["seed"]) for r in group if r.get("seed") is not None))
        row = {
            "manuscript_environment": env,
            "internal_environment": internal_env,
            "variant_key": variant_key,
            "variant_name": variant.get("name", variant_key),
            "episode_count": len(group),
            "success_count": len(success_records),
            "failure_count": len(group) - len(success_records),
            "collision_episode_count": sum(1 for r in success_records if r.get("collision_flag")),
            "collision_sample_count": sum(int(r.get("collision_count") or 0) for r in success_records),
            "seed_min": min(seeds) if seeds else "",
            "seed_max": max(seeds) if seeds else "",
        }
        row["collision_episode_rate"] = float(row["collision_episode_count"] / len(success_records)) if success_records else ""
        for field in metric_fields:
            stats = _stats_for_field(success_records, field)
            for stat_name, value in stats.items():
                row[f"{field}_{stat_name}"] = value
        rows.append(row)
    return rows


def _write_ablation_json(records, summary_rows, metadata, variants):
    by_environment = {}
    for record in records:
        env = record["manuscript_environment"]
        by_environment.setdefault(env, {"internal_environment": record["internal_environment"], "records": []})
        by_environment[env]["records"].append(record)
    payload = {"metadata": metadata, "variants": variants,
               "summary": summary_rows, "results_by_environment": by_environment}
    with open("ablation_results_by_environment.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    return "ablation_results_by_environment.json"


def _write_ablation_summary_csv(summary_rows):
    metric_fields = ["path_length", "maximum_curvature", "minimum_obstacle_clearance",
                     "smoothness", "wheel_effort", "centering", "route_runtime_seconds"]
    fieldnames = ["manuscript_environment", "internal_environment", "variant_key",
                  "variant_name", "episode_count", "success_count", "failure_count",
                  "collision_episode_count", "collision_episode_rate", "collision_sample_count",
                  "seed_min", "seed_max"]
    for field in metric_fields:
        for stat in ["mean", "std", "min", "max", "median"]:
            fieldnames.append(f"{field}_{stat}")
    _write_csv_rows("ablation_summary_all.csv", summary_rows, fieldnames)
    return "ablation_summary_all.csv"


def _latex_number(value, digits=3):
    if value == "" or value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "--"


def _write_ablation_latex(summary_rows):
    lines = [
        r"\begin{table}[!htbp]",
        r"\caption{Ablation summary for the proposed framework. Values are computed only from completed episodes; collision episodes indicate paths with at least one obstacle-cell intersection.}",
        r"\label{tab:ablation}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Environment & Variant & Length (m) & $\kappa_{max}$ & Min. clear. (m) & Coll. eps. & Runtime (s) \\",
        r"\midrule",
    ]
    for row in summary_rows:
        lines.append(
            f"{_latex_escape(row['manuscript_environment'])} & {_latex_escape(row['variant_name'])} & "
            f"{_latex_number(row.get('path_length_mean'))} & "
            f"{_latex_number(row.get('maximum_curvature_mean'))} & "
            f"{_latex_number(row.get('minimum_obstacle_clearance_mean'))} & "
            f"{row.get('collision_episode_count', 0)} & "
            f"{_latex_number(row.get('route_runtime_seconds_mean'))} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open("ablation_summary_for_latex.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "ablation_summary_for_latex.tex"


def _write_ablation_report(metadata, variants, summary_rows, generated_files):
    env_lines = ["- Easy -> Level-1", "- Moderate-I -> Level-2",
                 "- Moderate-III -> Level-3", "- Moderate-II -> Level-4",
                 "- Hard -> Level-5"]
    variant_lines = [f"- {v['name']} (`{v['key']}`): {v['code_level_meaning']}" for v in variants]
    collision_rows = [r for r in summary_rows if r.get("collision_episode_count", 0)]
    if collision_rows:
        collision_lines = [f"- {r['manuscript_environment']} / {r['variant_name']}: {r['collision_episode_count']} collision episode(s), {r['collision_sample_count']} obstacle-cell samples." for r in collision_rows]
    else:
        collision_lines = ["- No collision-producing variants in the completed ablation records."]
    report = f"""# Ablation Experiment Report

Generated: {metadata['generated_at']}

## Analysis Scope

- Ablation experiments for A* initialization, B-Spline representation, five-objective NSGA-II, softmin adaptive fusion, and post-processing.
- Module comparison, parameter-related verification, post-processing, and segmented local-optimum limitations.

## Command

```text
{metadata['command_used']}
```

## Ablation Variants

{chr(10).join(variant_lines)}

## Full Method Preservation

The default proposed method is preserved. Ablations are activated only through optional keyword arguments passed by `run_ablation_mode(...)` to `solve_patrol_nsga_all(...)` and `run_nsga_ii(...)`. Existing modes and default parameters remain unchanged.

## Episode Count

- Requested episodes per selected environment: {metadata['num_runs']}
- Completed record count: {metadata['record_count']}

## Seed Policy

Base seed: `{metadata['base_seed']}`. For each selected environment and episode, the seed is `base_seed + environment_index * 10000 + episode_index`. The same episode seed is reused across ablation variants to isolate module changes rather than random-seed changes.

## Environment-Name Mapping

{chr(10).join(env_lines)}

## Metrics Recorded

- path length
- maximum curvature
- minimum obstacle clearance
- smoothness
- wheel effort
- centering
- collision count and collision flag
- route runtime
- failure count and failure reason, if a variant fails

## Infeasible Or Collision-Producing Variants

{chr(10).join(collision_lines)}

## Limitations

- The `Reduced B-Spline representation` variant replaces B-Spline trajectory construction inside NSGA-II evaluation and final candidate construction with piecewise-linear interpolation. This is the closest controlled variant that keeps the planner executable. If safety repair is enabled, the post-processing pipeline may still use spline-based repair internally; this limitation must be reported honestly in the manuscript.
- The `No post-processing` variant intentionally disables collision repair, kinematic smoothing, obstacle escape, curvature re-splining, and junction smoothing. Collision or infeasible paths are recorded rather than hidden.
- These ablation outputs are experimental evidence only; no manuscript claims should be added until the requested local run has completed and the generated files have been inspected.

## Generated Files

{chr(10).join('- ' + name for name in generated_files)}
"""
    with open("ablation_experiment_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    return "ablation_experiment_report.md"


def _write_ablation_outputs(records, metadata, variants):
    summary_rows = _summarize_ablation_records(records)
    generated = []
    generated.append(_write_ablation_json(records, summary_rows, metadata, variants))
    generated.append(_write_ablation_summary_csv(summary_rows))
    generated.append(_write_ablation_latex(summary_rows))
    report_file = "ablation_experiment_report.md"
    generated.append(_write_ablation_report(metadata, variants, summary_rows, generated + [report_file]))
    return generated


def run_ablation_mode(num_runs=30, selected_env="all"):
    """Run controlled ablation variants for the proposed framework."""
    global grid, dist_field
    configs = _select_segment_envs(selected_env)
    variants = _ablation_variant_specs()
    records = []
    command_used = f"python -u patrollingAlgorithms.py ablation {int(num_runs)} {selected_env}"
    print(f"=== MODE: Ablation ({int(num_runs)} episode(s), env={selected_env}) ===")
    print("Ablation outputs will be written to ablation_* files only.")

    for cfg in configs:
        manuscript_env = cfg["manuscript_environment"]
        internal_env = cfg["internal_environment"]
        print(f"\n=== Ablation: {manuscript_env} ({internal_env}) ===")
        for run_idx in range(int(num_runs)):
            episode_index = run_idx + 1
            seed = ABLATION_BASE_SEED + cfg["index"] * 10000 + episode_index
            episode_id = f"{manuscript_env}_episode_{episode_index:03d}"
            for variant in variants:
                _set_timing_seed(seed)
                grid, waypoints = create_patrol_environment(internal_env)
                dist_field = ndi.distance_transform_edt(1 - grid)
                route_t0 = time.perf_counter()
                row = {
                    "manuscript_environment": manuscript_env,
                    "internal_environment": internal_env,
                    "source_environment": internal_env,
                    "environment_name": manuscript_env,
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "seed": seed,
                    "variant_key": variant["key"],
                    "variant_name": variant["name"],
                    "variant_kwargs": dict(variant["kwargs"]),
                    "status": "ok",
                    "failure_reason": "",
                }
                try:
                    paths = solve_patrol_nsga_all(waypoints, **variant["kwargs"])
                    path_adaptive = paths["Adaptive"]
                    row["route_runtime_seconds"] = float(time.perf_counter() - route_t0)
                    row.update(_compute_ablation_metrics(path_adaptive))
                    print(f"  ok | {episode_id} | {variant['name']} | seed={seed} | length={row['path_length']:.3f} | collisions={row['collision_count']} | runtime={row['route_runtime_seconds']:.3f}s")
                except Exception as exc:
                    row["status"] = "failed"
                    row["failure_reason"] = str(exc)
                    row["route_runtime_seconds"] = float(time.perf_counter() - route_t0)
                    for field in ["path_length", "maximum_curvature", "minimum_obstacle_clearance", "smoothness", "wheel_effort", "centering"]:
                        row[field] = None
                    row["collision_count"] = None
                    row["collision_flag"] = None
                    print(f"  failed | {episode_id} | {variant['name']} | seed={seed} | {exc}")
                records.append(row)

    metadata = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": "ablation",
        "command_used": command_used,
        "num_runs": int(num_runs),
        "selected_env": selected_env,
        "base_seed": ABLATION_BASE_SEED,
        "record_count": len(records),
        "environment_configs": configs,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor() or "not reported",
        "cpu_count": os.cpu_count() or "not reported",
    }
    generated = _write_ablation_outputs(records, metadata, variants)
    print("\nAblation mode complete. Generated files:")
    for filename in generated:
        print(f"  {filename}")
# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    # Required on Windows for multiprocessing (spawn) to work correctly
    multiprocessing.freeze_support()

    # Usage:
    #   python patrollingAlgorithms.py single            -> 1 epoch, path figures
    #   python patrollingAlgorithms.py test               -> 100 epochs, boxplot & raincloud
    #   python patrollingAlgorithms.py sensitivity [N] [all|rho_tau|rho|tau]
    #                                                   -> sensitivity analysis (N epochs, default 30)
    #   python patrollingAlgorithms.py segment_timing [N] [env|all] [all|proposed_only|competitors_only]
    #                                                   -> resumable segment-level timing
    #   python patrollingAlgorithms.py segment_timing_aggregate [all|proposed_only|competitors_only] [env|all]
    #                                                   -> rebuild timing summaries from checkpoints only
    #   python patrollingAlgorithms.py ablation [N] [env|all]
    #                                                   -> controlled ablation experiments
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "single"

    if mode == "single":
        print("=== MODE: Single Epoch (path figures) ===")
        run_single_epoch_mode()
    elif mode == "test":
        print("=== MODE: 100-Epoch Test (boxplot & raincloud) ===")
        run_test_mode(num_epochs=100)
    elif mode == "sensitivity":
        n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        scope_arg = sys.argv[3] if len(sys.argv) > 3 else "all"
        print(f"=== MODE: Sensitivity Analysis ({n_ep} epochs per setting, scope={scope_arg}) ===")
        run_sensitivity_mode(num_epochs=n_ep, scope=scope_arg)
    elif mode in {"segment_timing", "segment", "timing_segments"}:
        n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        env_arg = sys.argv[3] if len(sys.argv) > 3 else "all"
        method_arg = sys.argv[4] if len(sys.argv) > 4 else "all"
        print(f"=== MODE: Segment Timing ({n_runs} run(s), env={env_arg}, method_filter={method_arg}) ===")
        run_segment_timing_mode(num_runs=n_runs, selected_env=env_arg, method_filter=method_arg)
    elif mode in {"segment_timing_aggregate", "segment_aggregate", "timing_segments_aggregate"}:
        method_arg = sys.argv[2] if len(sys.argv) > 2 else "proposed_only"
        env_arg = sys.argv[3] if len(sys.argv) > 3 else "all"
        print(f"=== MODE: Segment Timing Aggregate (env={env_arg}, method_filter={method_arg}) ===")
        run_segment_timing_aggregate_mode(method_filter=method_arg, selected_env=env_arg)
    elif mode in {"ablation", "ablate"}:
        n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        env_arg = sys.argv[3] if len(sys.argv) > 3 else "all"
        run_ablation_mode(num_runs=n_runs, selected_env=env_arg)
    else:
        print(f"Unknown mode '{mode}'. Use 'single', 'test', 'sensitivity', 'segment_timing', 'segment_timing_aggregate', or 'ablation'.")
        sys.exit(1)



