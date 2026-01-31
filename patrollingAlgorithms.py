import numpy as np
import math
import random
import matplotlib.pyplot as plt
import heapq
import time
import json
from scipy import interpolate
import scipy.ndimage as ndi
from deap import base, creator, tools, algorithms

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
START_TANGENT_LEN = 4.0 

GRID_RES = 0.1
GRID_SIZE = 50

# Optimization resolution 
CHECK_SAMPLES = 200   
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

def get_max_curvature(path):
    if len(path) < 3: return 0.0
    dy = np.gradient(path[:, 0]); dx = np.gradient(path[:, 1])
    ddy = np.gradient(dy); ddx = np.gradient(dx)
    num = np.abs(dx * ddy - dy * ddx)
    den = np.power(dx**2 + dy**2, 1.5)
    den[den < 1e-9] = 1e-9 
    return np.max(num / den)

def get_curvature_penalty(path):
    """
    Gradient-based penalty for Non-Holonomic Constraints.
    """
    max_k = get_max_curvature(path)
    if max_k > MAX_CURVATURE:
        return (max_k - MAX_CURVATURE) * 5000.0
    return 0.0

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

def common_spline(control_points, num_samples=100):
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
        tck, u = interpolate.splprep([x, y], s=0, k=k)
        u_new = np.linspace(0, 1, num_samples)
        x_new, y_new = interpolate.splev(u_new, tck, der=0)
        return np.column_stack((y_new, x_new))
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
    interpolated_paths = []
    for ind in champion_inds:
        pts = common_spline(ind, num_samples=PLOT_SAMPLES)
        interpolated_paths.append(pts)
    min_len = min(len(p) for p in interpolated_paths)
    paths = [p[:min_len] for p in interpolated_paths]
    adaptive_points = []
    for t in range(0, min_len, 5): 
        candidates = [p[t] for p in paths]
        best_cand = candidates[0]; best_score = float('inf')
        for cand in candidates:
            iy, ix = int(round(cand[0])), int(round(cand[1]))
            iy = np.clip(iy, 0, GRID_SIZE-1); ix = np.clip(ix, 0, GRID_SIZE-1)
            dist = dist_field[iy, ix] * GRID_RES
            c_safe = 1.0 / (dist + 0.1)
            c_center = 1.0 / (dist + 0.01)
            score = 0.5*c_safe + 0.5*c_center
            if score < best_score:
                best_score = score; best_cand = cand
        adaptive_points.append(best_cand)
    return np.array(adaptive_points)

# ---------------------------------------------------------
# 3. HELPER ALGORITHMS (A* STAR)
# ---------------------------------------------------------
def a_star(current_grid, start, goal):
    safe_cells = int(np.ceil(MIN_SAFE_DIST / GRID_RES))
    inflated_grid = np.zeros_like(current_grid)
    inflated_grid[dist_field < safe_cells] = 1
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

# ---------------------------------------------------------
# 4. ALGORITHMS
# ---------------------------------------------------------

# --- 1. STANDARD GA ---
def run_ga_standard(start_node, goal_node, start_dir=None):
    if hasattr(creator, "FitnessGA"): del creator.FitnessGA
    if hasattr(creator, "IndividualGA"): del creator.IndividualGA
    creator.create("FitnessGA", base.Fitness, weights=(-1.0,))
    creator.create("IndividualGA", list, fitness=creator.FitnessGA)

    start_pts = [np.array(start_node)]
    if start_dir is not None:
        start_pts.append(np.array(start_node) + START_TANGENT_LEN * np.array(start_dir))

    gen_counter = [0]
    MAX_GENS = 40

    def eval_ga(individual):
        sorted_ind = sorted(individual, key=lambda p: np.linalg.norm(np.array(p) - np.array(start_node)))
        full_pts = start_pts + sorted_ind + [goal_node]
        path = common_spline(full_pts, CHECK_SAMPLES)
        length = get_path_length(path)
        safety = check_safety_vectorized(path)
        smooth = get_smoothness_cost(path)
        kinematic_penalty = get_curvature_penalty(path)
        cost = 5.0 * length + 1000.0 * safety + 50.0 * smooth + kinematic_penalty
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
            y = (1-alpha)*start_node[0] + alpha*goal_node[0] + random.uniform(-8,8)
            x = (1-alpha)*start_node[1] + alpha*goal_node[1] + random.uniform(-8,8)
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
    final_path = common_spline(start_pts + best + [goal_node], PLOT_SAMPLES)
    
    if len(final_path) > 10:
        end_dir = final_path[-1] - final_path[-10]
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
    print("  > Path 8: ACO Initialization...")
    path_nodes = run_aco_initialization(inflated_grid, start, goal)
    path = np.array(path_nodes, dtype=float)
    obs_array = np.array(obstacles_list)
    ITERATIONS = 50
    if len(path) < 5:
        ts = np.linspace(0, 1, 15)
        path = np.array(start) + np.outer(ts, np.array(goal)-np.array(start))
    current_best = np.copy(path)
    print("  > Path 8: WOA-APF Optimization...")
    for _ in range(ITERATIONS):
        new_path = np.copy(path)
        for i in range(1, len(path)-1):
            curr = path[i]
            dists = np.sqrt(np.sum((obs_array - curr)**2, axis=1))
            min_d = np.min(dists)
            f_rep = np.zeros(2)
            if min_d < 4.0:
                obs = obs_array[np.argmin(dists)]
                vec = curr - obs
                dist_v = np.linalg.norm(vec)
                if dist_v > 0: f_rep = (vec / dist_v) * 2.0 * (1.0/min_d)
            target = current_best[i]
            l = random.uniform(-1, 1)
            dist_t = np.linalg.norm(target - curr)
            spiral = dist_t * math.exp(1.0 * l) * math.cos(2*math.pi*l)
            smooth = ((path[i-1]+path[i+1])/2.0 - curr) * 0.5
            cand = curr + f_rep + (spiral * 0.1) + smooth
            if 0<=cand[0]<grid.shape[0] and 0<=cand[1]<grid.shape[1]:
                if grid[int(cand[0]), int(cand[1])] == 0: new_path[i] = cand
        path = new_path
        current_best = path
    return sparsify_path(path, min_dist=3.0)

def run_ab_woa(start_node, goal_node, start_dir=None):
    inflated_grid = np.zeros_like(grid)
    inflated_grid[dist_field < 2] = 1 
    obstacles_list = np.argwhere(grid == 1)
    sparse_path = solve_bio_hybrid_competitor(grid, inflated_grid, obstacles_list, start_node, goal_node)
    final_path = common_spline(sparse_path, PLOT_SAMPLES)
    if len(final_path) > 10:
        end_dir = final_path[-1] - final_path[-10]
        norm = np.linalg.norm(end_dir)
        end_dir = end_dir / norm if norm > 0 else np.array([0,1])
    else: end_dir = np.array([0,1])
    return final_path, end_dir

# --- 3. HWPSO (FIXED: PAPER EQ 20 + TUNED PENALTY) ---
def run_woa_pso(start_node, goal_node, start_dir=None):
    POP_SIZE, MAX_ITER, DIM = 40, 30, 7 
    X = np.zeros((POP_SIZE, DIM, 2))
    V = np.zeros((POP_SIZE, DIM, 2))
    
    # Initialization
    for i in range(POP_SIZE):
        for j in range(DIM):
            alpha = (j + 1) / (DIM + 1)
            y = (1-alpha)*start_node[0] + alpha*goal_node[0] + np.random.uniform(-10,10)
            x = (1-alpha)*start_node[1] + alpha*goal_node[1] + np.random.uniform(-10,10)
            X[i, j] = [np.clip(y,0,GRID_SIZE-1), np.clip(x,0,GRID_SIZE-1)]

    gbest = X[0].copy(); gbest_cost = float('inf')
    
    start_pts_list = [np.array(start_node)]
    if start_dir is not None:
        start_pts_list.append(np.array(start_node) + START_TANGENT_LEN * np.array(start_dir))
    start_pts = np.array(start_pts_list)

    def evaluate(ind):
        pts = ind[np.argsort(np.linalg.norm(ind - np.array(start_node), axis=1))]
        full_pts = np.vstack([start_pts, pts, goal_node])
        path = common_spline(full_pts, CHECK_SAMPLES)
        f_p = get_path_length(path)
        v_L = calculate_violation_paper(path) # Uses SUM, not MEAN
        
        # FIX: MU Increased to 10,000 to prevent collisions
        MU = 10000.0 
        cost = f_p * (1.0 + MU * v_L)
        
        cost += get_curvature_penalty(path)
        if start_dir is not None: cost += check_forward_motion(path, start_dir)
        return cost

    w = 0.6; c1 = 1.2; c2 = 1.2; V_MAX = 2.0 

    for t in range(MAX_ITER):
        a = 2 - 2 * t / MAX_ITER 
        current_iter_best = None; current_iter_best_cost = float('inf')
        for i in range(POP_SIZE):
            cost = evaluate(X[i])
            if cost < current_iter_best_cost:
                current_iter_best_cost = cost; current_iter_best = X[i].copy()
            if cost < gbest_cost:
                gbest_cost = cost; gbest = X[i].copy()
        
        Whale_Star = current_iter_best if current_iter_best is not None else gbest

        for i in range(POP_SIZE):
            r1 = np.random.rand(); r2 = np.random.rand()
            A = 2*a*r1 - a; C = 2*r2
            p = np.random.rand(); l = np.random.uniform(-1, 1); b = 1
            X_woa = X[i].copy()
            if p < 0.5:
                if abs(A) < 1: D = abs(C * Whale_Star - X[i]); X_woa = Whale_Star - A * D
                else: rand_idx = np.random.randint(0, POP_SIZE); D = abs(C * X[rand_idx] - X[i]); X_woa = X[rand_idx] - A * D
            else: D_prime = abs(Whale_Star - X[i]); X_woa = D_prime * np.exp(b*l) * np.cos(2*np.pi*l) + Whale_Star
            
            r1_pso = np.random.rand(); r2_pso = np.random.rand()
            V[i] = w * V[i] + c1 * r1_pso * (Whale_Star - X_woa) + c2 * r2_pso * (gbest - X_woa)
            V[i] = np.clip(V[i], -V_MAX, V_MAX)
            X[i] = np.clip(X_woa + V[i], 0, GRID_SIZE-1)

    pts = gbest[np.argsort(np.linalg.norm(gbest - np.array(start_node), axis=1))]
    final_path = common_spline(np.vstack([start_pts, pts, goal_node]), PLOT_SAMPLES)
    end_dir = final_path[-1] - final_path[-10]
    norm = np.linalg.norm(end_dir)
    end_dir = end_dir / norm if norm > 0 else np.array([0,1])
    return final_path, end_dir

# --- 5. NSGA-II (OPTIMIZED) ---
def run_nsga_ii(start_node, goal_node, start_dir=None):
    raw_path = a_star(grid, start_node, goal_node)
    if raw_path and len(raw_path) > 8:
        indices = np.linspace(0, len(raw_path)-1, 7, dtype=int)
        raw_seed = np.array(raw_path)[indices][1:-1].astype(float)
        seed_pts = [tuple(pt) for pt in raw_seed] if len(raw_seed) >= 2 else [tuple(start_node), tuple(goal_node)]
    else:
        seed_pts = [tuple(start_node), tuple(goal_node)]

    start_pts_fixed = [np.array(start_node)]
    if start_dir is not None:
        start_pts_fixed.append(np.array(start_node) + START_TANGENT_LEN * np.array(start_dir))

    if hasattr(creator, "FitnessNSGA"): del creator.FitnessNSGA
    if hasattr(creator, "IndividualNSGA"): del creator.IndividualNSGA
    creator.create("FitnessNSGA", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0))
    creator.create("IndividualNSGA", list, fitness=creator.FitnessNSGA)

    obstacles_list = np.argwhere(grid == 1)

    def eval_nsga(ind):
        full_pts = start_pts_fixed + list(ind) + [np.array(goal_node)]
        path = common_spline(full_pts, CHECK_SAMPLES)
        length = get_path_length(path)
        
        # Soft Kinematic Penalty
        max_k_penalty = get_curvature_penalty(path)
        
        effort = calculate_wheel_effort(path)
        center = calculate_centering_score(path, obstacles_list)
        
        min_x, min_y = np.min(path[:, 1]), np.min(path[:, 0])
        max_x, max_y = np.max(path[:, 1]), np.max(path[:, 0])
        safety_pen = 0
        if min_x < 0.5 or min_y < 0.5 or max_x > GRID_SIZE-0.5 or max_y > GRID_SIZE-0.5: safety_pen += 1e5
        
        iy = np.round(path[:, 0]).astype(int); ix = np.round(path[:, 1]).astype(int)
        iy = np.clip(iy, 0, GRID_SIZE - 1); ix = np.clip(ix, 0, GRID_SIZE - 1)
        dists = dist_field[iy, ix] * GRID_RES
        if np.any(dists < MIN_SAFE_DIST): safety_pen += 1e5
        
        if start_dir is not None: safety_pen += check_forward_motion(path, start_dir)
        if safety_pen > 10000: return (1e5, 1e5, 1e5, 1e5, 1e5)
        
        return (length, max_k_penalty, effort, center, safety_pen)

    def mut_nsga(ind, indpb=0.2):
        for i in range(len(ind)):
            if random.random() < indpb:
                ind[i] = (np.clip(ind[i][0] + random.gauss(0, 3.0), 0, GRID_SIZE-1),
                          np.clip(ind[i][1] + random.gauss(0, 3.0), 0, GRID_SIZE-1))
        return ind,

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.IndividualNSGA([pt for pt in seed_pts]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_nsga)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mut_nsga, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=40)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=40, lambda_=40, cxpb=0.7, mutpb=0.3, ngen=25, verbose=False)
    
    best_len = min(pop, key=lambda x: x.fitness.values[0])
    best_curv = min(pop, key=lambda x: x.fitness.values[1])
    best_effort = min(pop, key=lambda x: x.fitness.values[2])
    best_center = min(pop, key=lambda x: x.fitness.values[3])
    best_safe = min(pop, key=lambda x: x.fitness.values[4])
    
    champions = [
        start_pts_fixed + list(best_len) + [np.array(goal_node)],
        start_pts_fixed + list(best_curv) + [np.array(goal_node)],
        start_pts_fixed + list(best_effort) + [np.array(goal_node)],
        start_pts_fixed + list(best_center) + [np.array(goal_node)],
        start_pts_fixed + list(best_safe) + [np.array(goal_node)]
    ]
    
    adaptive_coords = construct_adaptive_path(champions, obstacles_list)
    resampled = adaptive_coords[::10]
    if len(resampled) > 2:
        resampled[0] = adaptive_coords[0]; resampled[-1] = adaptive_coords[-1]
        path_adaptive = common_spline(resampled, PLOT_SAMPLES)
    else:
        path_adaptive = common_spline(champions[0], PLOT_SAMPLES)

    paths = {
        "Length": common_spline(champions[0], PLOT_SAMPLES),
        "Smooth": common_spline(champions[1], PLOT_SAMPLES),
        "Effort": common_spline(champions[2], PLOT_SAMPLES),
        "Centered": common_spline(champions[3], PLOT_SAMPLES),
        "Safe": common_spline(champions[4], PLOT_SAMPLES),
        "Adaptive": path_adaptive
    }
    
    end_dir = path_adaptive[-1] - path_adaptive[-10]
    norm = np.linalg.norm(end_dir)
    end_dir = end_dir / norm if norm > 0 else np.array([0,1])
    return paths, end_dir

# ---------------------------------------------------------
# 6. EXECUTION & PLOTTING
# ---------------------------------------------------------
def create_patrol_environment(difficulty):
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    waypoints = [(5, 5), (5, 45), (45, 45), (45, 5)] 
    
    if difficulty == "Easy":
        g[2:8, 22:28] = 1   
        g[42:48, 22:28] = 1 
    elif difficulty == "Moderate":
        g[2:8, 22:28] = 1   
        g[22:28, 42:48] = 1 
        g[42:48, 22:28] = 1 
        g[22:28, 2:8] = 1   
    elif difficulty == "Hard":
        g[0:10, 20:30] = 1   
        g[20:30, 40:50] = 1  
        g[40:50, 20:30] = 1  
        g[20:30, 0:10] = 1   
        g[22:28, 22:28] = 1  
        
    return g, waypoints

def solve_patrol_single(algo_func, waypoints):
    full_path = []
    targets = waypoints.copy()
    targets.append(waypoints[0]) 
    curr_dir = FIXED_START_DIR.copy()
    for i in range(len(targets) - 1):
        start, end = targets[i], targets[i+1]
        segment, end_dir = algo_func(start, end, start_dir=curr_dir)
        curr_dir = end_dir
        if len(full_path) == 0: full_path = segment
        else: full_path = np.vstack((full_path, segment[1:]))
    return full_path

def solve_patrol_nsga_all(waypoints):
    full_paths = {k: [] for k in ["Length", "Smooth", "Effort", "Centered", "Safe", "Adaptive"]}
    targets = waypoints.copy()
    targets.append(waypoints[0])
    curr_dir = FIXED_START_DIR.copy()
    for i in range(len(targets) - 1):
        start, end = targets[i], targets[i+1]
        seg_dict, end_dir = run_nsga_ii(start, end, start_dir=curr_dir)
        curr_dir = end_dir
        for k in full_paths:
            if len(full_paths[k]) == 0: full_paths[k] = seg_dict[k]
            else: full_paths[k] = np.vstack((full_paths[k], seg_dict[k][1:]))
    return full_paths

if __name__ == "__main__":
    seedno=50
    random.seed(seedno); np.random.seed(seedno)
    ENV_DIFFICULTIES = ["Easy", "Moderate", "Hard"] 
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
        
        with open(f"calculation_times_{diff}.json", "w") as f:
            json.dump(timing_results, f, indent=4)
        print(f"Timing saved to calculation_times_{diff}.json")

        plt.figure(figsize=(12, 12))
        plt.imshow(grid, cmap='Greys', origin='lower')
        wx, wy = zip(*waypoints)
        plt.scatter(wy, wx, c='red', s=150, marker='X', zorder=20)
        
        def plot_with_arrows(path, c, s, w, base_label):
            length = get_path_length(path)
            label = f"{base_label} ({length:.2f}m)"
            plt.plot(path[:, 1], path[:, 0], color=c, linestyle=s, linewidth=w, label=label, alpha=0.8)
            
            arrow_indices = [int(len(path) * 0.125), int(len(path) * 0.375), 
                             int(len(path) * 0.625), int(len(path) * 0.875)]
            
            for i in arrow_indices:
                p_curr = path[i]
                p_next = path[i+5] 
                dx = p_next[1] - p_curr[1]
                dy = p_next[0] - p_curr[0]
                plt.arrow(p_curr[1], p_curr[0], dx*0.1, dy*0.1, shape='full', lw=0, 
                          length_includes_head=True, head_width=0.8, color=c, zorder=25)

        plot_with_arrows(p_nsga["Length"], 'blue', ':', 1.5, '1. NSGA-II Length')
        plot_with_arrows(p_nsga["Smooth"], 'lime', '--', 1.5, '2. NSGA-II Smooth')
        plot_with_arrows(p_nsga["Effort"], 'pink', '--', 1.5, '3. NSGA-II Effort')
        plot_with_arrows(p_nsga["Centered"], 'cyan', '--', 1.5, '4. NSGA-II Centered')
        plot_with_arrows(p_nsga["Safe"], 'magenta', '--', 1.5, '5. NSGA-II Safe')
        
        plot_with_arrows(p_ga, 'gray', '-', 2.5, '7. Standard GA')
        plot_with_arrows(p_ab, 'red', '-', 2.5, '8. AB-WOA-APF')
        plot_with_arrows(p_wp, 'purple', '-', 2.5, '9. HWPSO')
        
        plot_with_arrows(p_nsga["Adaptive"], 'gold', '-', 4.0, '6. NSGA-II Adaptive')

        plt.legend(loc='upper right', fontsize='small')
        plt.title(f"Comparison of 9 Path Planning Strategies ({diff})")
        plt.tight_layout()
        plt.savefig(f"Patrolling_9Paths_{diff}.png")
        plt.close()
        print(f"Saved Patrolling_9Paths_{diff}.png")