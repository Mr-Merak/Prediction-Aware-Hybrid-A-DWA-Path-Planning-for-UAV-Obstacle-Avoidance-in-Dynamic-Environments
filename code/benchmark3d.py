"""
benchmark3d.py -- Preliminary 3D extension (Reviewer 2 suggestion #4; Reviewer 1 #1).

Extends the core mechanisms of the 2D framework to three dimensions:
  * 3D constant-velocity Kalman filter (state [x,y,z,vx,vy,vz]),
  * 3D Gaussian risk field (covariance from the predicted position covariance),
  * 3D risk-adaptive A* (6-neighbour voxel search),
  * a 3D local planner over (v, omega, v_z), optionally prediction-aware.

A single altitude-varying scenario is evaluated: a low ridge blocks the direct
route below z = 24 m and a high slab blocks it above z = 30 m, so the UAV must
climb into the [24, 30] m window, pass through, and descend -- behaviour that is
impossible to express in the 2D formulation. Dynamic obstacles move in 3D.

Configurations:
    astar3d          : 3D risk-adaptive A* + pure-pursuit tracking (no local layer)
    astar3d_dwa      : 3D A* + static-obstacle DWA
    astar3d_pred_dwa : 3D A* + prediction-aware DWA

Run:
    py benchmark3d.py --runs 50 --jobs 7
"""
import argparse
import heapq
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
FIG_DIR = os.path.dirname(HERE)

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
BOUNDS = (0.0, 100.0, 0.0, 100.0, 0.0, 40.0)   # x0,x1,y0,y1,z0,z1
RES = 2.0
GRID = (50, 50, 20)
START = (5.0, 50.0, 15.0)
GOAL = (95.0, 50.0, 15.0)
UAV_R = 1.0
V_MAX = 2.5
OMEGA_MAX = 0.9
VZ_MAX = 2.0
DETECT_R = 20.0
DT = 0.2
REPLAN = 5.0
MAX_T = 200.0
GOAL_TOL = 4.0
MARGIN = 0.5
N_PRED = 15            # prediction steps (3 s)
BETA = 10.0
R0 = 0.6
Q_PROC = 0.5
R_OBS = 2.0

CONFIGS = ["astar3d", "astar3d_dwa", "astar3d_pred_dwa"]


# --------------------------------------------------------------------------
# 3D Kalman filter (constant velocity)
# --------------------------------------------------------------------------
class KF3D:
    def __init__(self, dt=DT):
        self.dt = dt
        self.F = np.eye(6)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.Q = np.eye(6) * Q_PROC
        for i in range(3):
            self.Q[i, i] = Q_PROC * dt ** 4 / 4
            self.Q[i + 3, i + 3] = Q_PROC * dt ** 2
        self.R = np.eye(3) * R_OBS
        self.x = np.zeros(6)
        self.P = np.eye(6) * 5.0

    def init(self, p):
        self.x[:3] = p
        self.x[3:] = 0.0

    def update(self, z):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = np.asarray(z) - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def predict(self, k):
        return (np.linalg.matrix_power(self.F, k) @ self.x)[:3].copy()


# --------------------------------------------------------------------------
# Dynamic obstacle
# --------------------------------------------------------------------------
class DynObstacle:
    def __init__(self, x, y, z, r, vx, vy, vz):
        self.x, self.y, self.z = x, y, z
        self.r = r
        self.vx, self.vy, self.vz = vx, vy, vz

    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        x0, x1, y0, y1, z0, z1 = BOUNDS
        if self.x < x0 + self.r:
            self.x = x0 + self.r; self.vx = abs(self.vx)
        elif self.x > x1 - self.r:
            self.x = x1 - self.r; self.vx = -abs(self.vx)
        if self.y < y0 + self.r:
            self.y = y0 + self.r; self.vy = abs(self.vy)
        elif self.y > y1 - self.r:
            self.y = y1 - self.r; self.vy = -abs(self.vy)
        if self.z < z0 + self.r:
            self.z = z0 + self.r; self.vz = abs(self.vz)
        elif self.z > z1 - self.r:
            self.z = z1 - self.r; self.vz = -abs(self.vz)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class Env3D:
    def __init__(self):
        self.static = []          # (x, y, z, r)
        self.dynamic = []         # DynObstacle
        self.kfs = []             # KF3D per dynamic obstacle
        self.risk = np.zeros(GRID)
        self.static_grid = np.zeros(GRID, dtype=bool)

    def add_static(self, x, y, z, r):
        self.static.append((x, y, z, r))

    def build_static_grid(self):
        """Precompute the dilated static-obstacle occupancy grid (O(1) lookups)."""
        self.static_grid[:] = False
        for (sx, sy, sz, sr) in self.static:
            rad = sr + UAV_R + MARGIN
            ix0 = max(0, int((sx - rad) / RES)); ix1 = min(GRID[0], int((sx + rad) / RES) + 1)
            iy0 = max(0, int((sy - rad) / RES)); iy1 = min(GRID[1], int((sy + rad) / RES) + 1)
            iz0 = max(0, int((sz - rad) / RES)); iz1 = min(GRID[2], int((sz + rad) / RES) + 1)
            for ix in range(ix0, ix1):
                for iy in range(iy0, iy1):
                    for iz in range(iz0, iz1):
                        cx = ix * RES + RES / 2
                        cy = iy * RES + RES / 2
                        cz = iz * RES + RES / 2
                        if (cx - sx) ** 2 + (cy - sy) ** 2 + (cz - sz) ** 2 < rad ** 2:
                            self.static_grid[ix, iy, iz] = True

    def add_dynamic(self, x, y, z, r, vx, vy, vz):
        o = DynObstacle(x, y, z, r, vx, vy, vz)
        self.dynamic.append(o)
        kf = KF3D()
        kf.init((x, y, z))
        self.kfs.append(kf)

    def update_kf(self):
        for o, kf in zip(self.dynamic, self.kfs):
            kf.update((o.x, o.y, o.z))

    def step_dynamic(self):
        for o in self.dynamic:
            o.step(DT)

    # ---- risk field (3D Gaussian, max over obstacles and steps) ----
    def compute_risk(self, use_kf=True, n_steps=N_PRED):
        g = np.zeros(GRID)
        x0, x1, y0, y1, z0, z1 = BOUNDS
        xs = np.linspace(x0 + RES / 2, x1 - RES / 2, GRID[0])
        ys = np.linspace(y0 + RES / 2, y1 - RES / 2, GRID[1])
        zs = np.linspace(z0 + RES / 2, z1 - RES / 2, GRID[2])
        for oi, o in enumerate(self.dynamic):
            kf = self.kfs[oi]
            for step in range(1, n_steps + 1):
                if use_kf:
                    px, py, pz = kf.predict(step)
                    cov = kf.P[:3, :3]
                else:
                    px = o.x + o.vx * step * DT
                    py = o.y + o.vy * step * DT
                    pz = o.z + o.vz * step * DT
                    cov = np.eye(3) * (1.0 + step * 0.5)
                cov = cov + np.eye(3) * o.r ** 2
                # local block around predicted position
                spread = max(2, int(math.sqrt(cov[0, 0] + cov[1, 1] + cov[2, 2]) / RES) + 1)
                ix0 = max(0, int(px / RES) - spread); ix1 = min(GRID[0], int(px / RES) + spread + 1)
                iy0 = max(0, int(py / RES) - spread); iy1 = min(GRID[1], int(py / RES) + spread + 1)
                iz0 = max(0, int(pz / RES) - spread); iz1 = min(GRID[2], int(pz / RES) + spread + 1)
                if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
                    continue
                X = xs[ix0:ix1, None, None] - px
                Y = ys[iy0:iy1][None, :, None] - py
                Z = zs[iz0:iz1][None, None, :] - pz
                try:
                    si = np.linalg.inv(cov)
                    maha = (si[0, 0] * X * X + si[1, 1] * Y * Y + si[2, 2] * Z * Z
                            + (si[0, 1] + si[1, 0]) * X * Y
                            + (si[0, 2] + si[2, 0]) * X * Z
                            + (si[1, 2] + si[2, 1]) * Y * Z)
                    risk = np.exp(-maha / 2.0)
                except np.linalg.LinAlgError:
                    sv = 2.0 * (cov[0, 0] + cov[1, 1] + cov[2, 2]) + 1e-6
                    risk = np.exp(-(X * X + Y * Y + Z * Z) / sv)
                np.maximum(g[ix0:ix1, iy0:iy1, iz0:iz1], risk,
                           out=g[ix0:ix1, iy0:iy1, iz0:iz1])
        np.clip(g, 0.0, 1.0, out=g)
        self.risk = g

    def risk_at(self, x, y, z):
        ix, iy, iz = int(x / RES), int(y / RES), int(z / RES)
        if 0 <= ix < GRID[0] and 0 <= iy < GRID[1] and 0 <= iz < GRID[2]:
            return float(self.risk[ix, iy, iz])
        return 0.0

    def local_risk(self, x, y, z, w=2):
        s, n = 0.0, 0
        for dx in range(-w, w + 1):
            for dy in range(-w, w + 1):
                for dz in range(-w, w + 1):
                    s += self.risk_at(x + dx * RES, y + dy * RES, z + dz * RES)
                    n += 1
        return s / n

    def is_collision(self, x, y, z, r, margin=MARGIN):
        for (sx, sy, sz, sr) in self.static:
            if math.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2) < r + sr + margin:
                return True
        for o in self.dynamic:
            if math.sqrt((x - o.x) ** 2 + (y - o.y) ** 2 + (z - o.z) ** 2) < r + o.r + margin:
                return True
        return False

    def static_collision(self, x, y, z, r, margin=MARGIN):
        for (sx, sy, sz, sr) in self.static:
            if math.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2) < r + sr + margin:
                return True
        return False

    def static_collision_grid(self, x, y, z):
        """O(1) dilated static-occupancy lookup for the DWA forward simulation."""
        ix, iy, iz = int(x / RES), int(y / RES), int(z / RES)
        if 0 <= ix < GRID[0] and 0 <= iy < GRID[1] and 0 <= iz < GRID[2]:
            return bool(self.static_grid[ix, iy, iz])
        return True


# --------------------------------------------------------------------------
# 3D risk-adaptive A*
# --------------------------------------------------------------------------
class AStar3D:
    def __init__(self, env, start, goal, beta=BETA, r0=R0):
        self.env = env
        self.start = self._snap(start)
        self.goal = self._snap(goal)
        self.beta = beta
        self.r0 = r0
        self.dmax = math.sqrt((BOUNDS[1] - BOUNDS[0]) ** 2 + (BOUNDS[3] - BOUNDS[2]) ** 2
                              + (BOUNDS[5] - BOUNDS[4]) ** 2)

    def _snap(self, p):
        # Grid cell i covers [i*RES, (i+1)*RES) with centre i*RES + RES/2, so a
        # world coordinate maps to floor(p/RES) -- round() would shift half-integer
        # coordinates (e.g. z=15 with RES=2) by one cell.
        return (int(p[0] / RES), int(p[1] / RES), int(p[2] / RES))

    def _h_dist(self, n):
        g = self.goal
        return math.sqrt((n[0] - g[0]) ** 2 + (n[1] - g[1]) ** 2 + (n[2] - g[2]) ** 2) * RES

    def _risk(self, n):
        return self.env.risk[n[0], n[1], n[2]]

    def _alpha(self, rbar):
        return 1.0 / (1.0 + math.exp(-self.beta * (rbar - self.r0)))

    def _neighbors(self, n):
        out = []
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nx, ny, nz = n[0] + dx, n[1] + dy, n[2] + dz
            if not (0 <= nx < GRID[0] and 0 <= ny < GRID[1] and 0 <= nz < GRID[2]):
                continue
            if self.env.static_grid[nx, ny, nz]:
                continue
            out.append((nx, ny, nz))
        return out

    def run(self):
        start, goal = self.start, self.goal
        openq = [(0.0, start)]
        gscore = {start: 0.0}
        came = {}
        closed = set()
        while openq:
            f, cur = heapq.heappop(openq)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return [((p[0] * RES + RES / 2), (p[1] * RES + RES / 2), (p[2] * RES + RES / 2))
                        for p in path]
            # local mean risk over 3x3x3
            s, n = 0.0, 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        q = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                        if 0 <= q[0] < GRID[0] and 0 <= q[1] < GRID[1] and 0 <= q[2] < GRID[2]:
                            s += self._risk(q); n += 1
            rbar = s / n if n else 0.0
            alpha = self._alpha(rbar)
            h_risk = rbar * self.dmax
            h_dist = self._h_dist(cur)
            h = (1 - alpha) * h_dist + alpha * h_risk
            for nb in self._neighbors(cur):
                if nb in closed:
                    continue
                step = math.sqrt((nb[0] - cur[0]) ** 2 + (nb[1] - cur[1]) ** 2
                                 + (nb[2] - cur[2]) ** 2) * RES
                gnew = gscore[cur] + step
                if gnew < gscore.get(nb, float("inf")):
                    gscore[nb] = gnew
                    came[nb] = cur
                    heapq.heappush(openq, (gnew + h, nb))
        return None


# --------------------------------------------------------------------------
# 3D DWA (prediction-aware option)
# --------------------------------------------------------------------------
class DWA3D:
    def __init__(self, env, uav, goal, pred=False):
        self.env = env
        self.uav = uav
        self.goal = goal
        self.pred = pred
        self.predict_time = 1.5
        self.dmax = math.sqrt((BOUNDS[1] - BOUNDS[0]) ** 2 + (BOUNDS[3] - BOUNDS[2]) ** 2
                              + (BOUNDS[5] - BOUNDS[4]) ** 2)

    def _simulate(self, v, omega, vz, dyn_preds):
        x, y, z = self.uav.x, self.uav.y, self.uav.z
        heading = self.uav.heading
        horizon = int(self.predict_time / DT)
        for k in range(1, horizon + 1):
            x += v * math.cos(heading) * DT
            y += v * math.sin(heading) * DT
            z += vz * DT
            heading += omega * DT
            if self.env.static_collision_grid(x, y, z):
                return None
            for oi in dyn_preds:
                px, py, pz = dyn_preds[oi][k - 1]
                if math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) < UAV_R + self.env.dynamic[oi].r + MARGIN:
                    return None
        return (x, y, z, heading)

    def _dyn_preds(self):
        horizon = int(self.predict_time / DT)
        preds = {}
        for oi, o in enumerate(self.env.dynamic):
            lst = []
            for k in range(1, horizon + 1):
                if self.pred:
                    lst.append(tuple(self.env.kfs[oi].predict(k)))
                else:
                    lst.append((o.x, o.y, o.z))
            preds[oi] = lst
        return preds

    def compute(self, path):
        dyn_preds = self._dyn_preds()
        # steer toward a look-ahead point on the path (inherits its altitude profile)
        target = find_lookahead3d(path, self.uav, 4.0, 20)
        tx, ty, tz = target
        v_cand = np.linspace(0.5, V_MAX * 0.8, 6)
        om_cand = np.linspace(-OMEGA_MAX, OMEGA_MAX, 11)
        vz_cand = np.linspace(-VZ_MAX, VZ_MAX, 5)
        best = None
        best_score = -1e18
        for v in v_cand:
            for om in om_cand:
                for vz in vz_cand:
                    end = self._simulate(v, om, vz, dyn_preds)
                    if end is None:
                        continue
                    ex, ey, ez, eh = end
                    dx, dy, dz = tx - ex, ty - ey, tz - ez
                    dg = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dg < 1e-6:
                        head_score = 1.0
                    else:
                        ux, uy = math.cos(eh), math.sin(eh)
                        vzn = vz / max(abs(vz), 1e-6)  # sign of vertical intent
                        uu = math.sqrt(ux * ux + uy * uy + vzn * vzn)
                        head_score = (ux * dx + uy * dy + vzn * dz) / (dg * uu)
                    score = 1.0 * head_score - 0.3 * dg / self.dmax
                    if score > best_score:
                        best_score = score
                        best = (float(v), float(om), float(vz))
        return best if best is not None else (0.5, 0.0, 0.0)


# --------------------------------------------------------------------------
# UAV
# --------------------------------------------------------------------------
class UAV3D:
    def __init__(self, p):
        self.x, self.y, self.z = p
        self.heading = math.atan2(GOAL[1] - p[1], GOAL[0] - p[0])
        self.v, self.omega, self.vz = 0.0, 0.0, 0.0

    def update(self, v, omega, vz, dt):
        self.v = v
        self.omega = omega
        self.vz = vz
        self.heading += omega * dt
        self.x += v * math.cos(self.heading) * dt
        self.y += v * math.sin(self.heading) * dt
        self.z = max(BOUNDS[4], min(BOUNDS[5], self.z + vz * dt))

    def dist(self, p):
        return math.sqrt((self.x - p[0]) ** 2 + (self.y - p[1]) ** 2 + (self.z - p[2]) ** 2)


# --------------------------------------------------------------------------
# Scenario: low ridge (block below z=24) + high slab (block above z=30)
# --------------------------------------------------------------------------
def build_env(seed):
    env = Env3D()
    # A full-spanning wall in the y-z plane at x=50:
    #   low ridge blocks z < ~23 (must climb over it),
    #   high slab blocks z > ~28.5 (must stay below it),
    # leaving a free window at z ~ [23.5, 28.5] that the UAV must climb into.
    rng = np.random.default_rng(seed)
    for yy in np.arange(-4.0, 104.0, 2.5):
        for zz in np.arange(2.0, 21.0, 3.0):
            env.add_static(50.0, yy, zz, 2.0)     # ridge (low, blocks z < ~23.5)
        for zz in np.arange(32.0, 41.0, 3.0):
            env.add_static(50.0, yy, zz, 2.0)     # slab (high, blocks z > ~28.5)
    # dynamic obstacles in 3D (horizontal, constant velocity)
    for _ in range(3):
        x = float(rng.uniform(10, 90)); y = float(rng.uniform(10, 90))
        z = float(rng.uniform(5, 35)); r = 2.0
        ang = rng.uniform(0, 2 * math.pi); sp = rng.uniform(0.8, 1.2)
        env.add_dynamic(x, y, z, r, sp * math.cos(ang), sp * math.sin(ang), 0.0)
    env.build_static_grid()
    return env


def find_lookahead3d(path, uav, min_dist, max_off):
    if not path:
        return GOAL
    best_i, best_d = 0, 1e18
    for i, p in enumerate(path):
        d = math.sqrt((uav.x - p[0]) ** 2 + (uav.y - p[1]) ** 2 + (uav.z - p[2]) ** 2)
        if d < best_d:
            best_d, best_i = d, i
    for s in range(1, max_off + 1):
        idx = min(best_i + s, len(path) - 1)
        p = path[idx]
        if math.sqrt((uav.x - p[0]) ** 2 + (uav.y - p[1]) ** 2 + (uav.z - p[2]) ** 2) >= min_dist:
            return p
    return path[-1]


def run_episode(cfg, seed):
    env = build_env(seed)
    uav = UAV3D(START)
    dwa = DWA3D(env, uav, GOAL, pred=(cfg == "astar3d_pred_dwa"))
    use_dwa = cfg in ("astar3d_dwa", "astar3d_pred_dwa")

    sim_time = 0.0
    env.compute_risk(use_kf=True)
    planner = AStar3D(env, START, GOAL)
    path = planner.run()
    last_replan = 0.0
    outcome = "timeout"
    traj = [(uav.x, uav.y, uav.z)]

    n_steps = int(MAX_T / DT)
    for _ in range(n_steps):
        sim_time += DT
        env.step_dynamic()
        env.update_kf()

        if env.is_collision(uav.x, uav.y, uav.z, UAV_R):
            outcome = "collision"
            break
        if uav.dist(GOAL) < GOAL_TOL:
            outcome = "success"
            break

        if (sim_time - last_replan) >= REPLAN:
            env.compute_risk(use_kf=True)          # risk field only needed at replan
            planner = AStar3D(env, (uav.x, uav.y, uav.z), GOAL)
            path = planner.run() or path
            last_replan = sim_time

        if path is None:
            path = planner.run() or [GOAL]

        if use_dwa:
            vel = dwa.compute(path)
            v, omega, vz = vel
        else:
            target = find_lookahead3d(path, uav, 6.0, 30)
            dx, dy, dz = target[0] - uav.x, target[1] - uav.y, target[2] - uav.z
            dhor = math.sqrt(dx * dx + dy * dy)
            desired_heading = math.atan2(dy, dx)
            dhead = desired_heading - uav.heading
            dhead = math.atan2(math.sin(dhead), math.cos(dhead))
            v = V_MAX * 0.8
            omega = max(-OMEGA_MAX, min(OMEGA_MAX, 2.0 * dhead))
            vz = max(-VZ_MAX, min(VZ_MAX, dz * 1.0))

        uav.update(v, omega, vz, DT)
        traj.append((uav.x, uav.y, uav.z))

    return dict(config=cfg, seed=seed, outcome=outcome, traj=traj)


def _worker(job):
    cfg, seed = job
    try:
        return run_episode(cfg, seed)
    except Exception as exc:
        return dict(config=cfg, seed=seed, outcome="error:%s" % exc, traj=[])


def wilson_ci(succ, n):
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = succ / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def write_xlsx(rows, path):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Scenario", "Configuration", "Runs", "Success Rate (%)",
               "95% CI Low", "95% CI High"])
    for cfg in CONFIGS:
        rs = [r for r in rows if r["config"] == cfg]
        n = len(rs)
        succ = sum(1 for r in rs if r["outcome"] == "success")
        lo, hi = wilson_ci(succ, n)
        ws.append(["3D altitude-varying", cfg, n, round(100.0 * succ / n, 1),
                   round(100.0 * lo, 1), round(100.0 * hi, 1)])
    ws2 = wb.create_sheet("Raw runs")
    ws2.append(["Configuration", "Seed", "Outcome"])
    for r in rows:
        ws2.append([r["config"], r["seed"], r["outcome"]])
    wb.save(path)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    jobs = [(c, s) for c in CONFIGS for s in range(args.runs)]
    print("3D preliminary extension: %d runs x %d configs | %d workers"
          % (args.runs, len(CONFIGS), args.jobs))
    t0 = time.time()
    rows = []
    with mp.Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs), 1):
            rows.append(r)
            if i % 30 == 0 or i == len(jobs):
                print("  %d/%d (%.0f s)" % (i, len(jobs), time.time() - t0), flush=True)

    print("\n===== 3D altitude-varying scenario =====")
    print("%-22s %8s %-16s" % ("Configuration", "Succ%", "95% CI"))
    for cfg in CONFIGS:
        rs = [r for r in rows if r["config"] == cfg]
        n = len(rs)
        succ = sum(1 for r in rs if r["outcome"] == "success")
        lo, hi = wilson_ci(succ, n)
        print("  %-20s %6.1f   [%4.1f, %4.1f]  (%d/%d)"
              % (cfg, 100.0 * succ / n, 100.0 * lo, 100.0 * hi, succ, n))

    xlsx_path = os.path.join(DATA_DIR, "experiment_data_3d.xlsx")
    write_xlsx(rows, xlsx_path)
    print("\nWrote %s" % xlsx_path)

    # figure: one representative 3D trajectory per config
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 4.5))
    for ci, cfg in enumerate(CONFIGS):
        ax = fig.add_subplot(1, 3, ci + 1, projection="3d")
        rs = [r for r in rows if r["config"] == cfg]
        r = next((x for x in rs if x["outcome"] == "success"), rs[0])
        tr = r["traj"]
        xs = [p[0] for p in tr]; ys = [p[1] for p in tr]; zs = [p[2] for p in tr]
        ax.plot(xs, ys, zs, color="C0", lw=2)
        ax.scatter([START[0]], [START[1]], [START[2]], color="C0", s=40)
        ax.scatter([GOAL[0]], [GOAL[1]], [GOAL[2]], color="C3", marker="x", s=60)
        ax.set_title("%s (%s)" % (cfg, r["outcome"]), fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_zlim(0, 40)
    fig.suptitle("3D preliminary extension: altitude-varying obstacle scenario")
    fig_path = os.path.join(FIG_DIR, "figure3d.png")
    fig.savefig(fig_path, dpi=120)
    print("\nWrote %s" % fig_path)


if __name__ == "__main__":
    main()
