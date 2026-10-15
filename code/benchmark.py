"""
benchmark.py -- Batch simulation harness for the five configurations reported in
the paper (Tables 1 and 2).

It imports the algorithm classes from 11.py *without modifying that file* and
re-runs the simulation loop of main() headlessly, parameterised by configuration.

Configurations
--------------
  astar      Traditional A*      : Euclidean heuristic, risk field NOT used by the
                                   planner, no smoothing, no DWA, pure-pursuit
                                   tracking, replan every 5 s.        (paper 5.2-1)
  ia_only    Improved A* w/o DWA : risk-adaptive heuristic + smoothing, pure
                                   pursuit, no DWA.                   (paper 5.2-2)
  proposed   IA* + DWA           : full framework.                    (paper 5.2-3)
  abl_kf     w/o Kalman filter   : risk field built from constant-velocity
                                   extrapolation with fixed diagonal covariance
                                   Sigma = I*(1.0 + step*0.5).        (paper 5.3-1)
  abl_adah   w/o adaptive heur.  : A* degraded to pure Euclidean distance; KF and
                                   risk field retained for DWA collision checking
                                   and mode switching.                (paper 5.3-2)

Methodology
-----------
* Paired seeds: for a given seed every configuration sees an identical
  environment, so the comparison is matched rather than independent.
* Per paper Sec. 5.2, only the *initial positions* of the dynamic obstacles are
  randomised; radii and velocity vectors are held at their nominal values.
* Collision criterion (paper Sec. 5.2): min distance to any obstacle <
  uav.radius + obs.radius + 0.5 m.
* Trajectory-quality metrics (path length, mean local risk, peak angular
  velocity utilisation) are aggregated over SUCCESSFUL runs only. Averaging a
  run that crashed after 20 m into the path-length statistic would flatter the
  weakest planner, so failed runs are excluded from those three metrics.
* Replanning time is aggregated over ALL runs -- it is a computational cost that
  is incurred whether or not the flight ultimately succeeds. Timing brackets the
  global replan only (A* search + redundancy removal + spline); DWA and all
  plotting are excluded, per paper Sec. 5.2.
* The risk value used for *scoring* is always the Kalman-based field, for every
  configuration including abl_kf and astar. The risk field is the measuring
  instrument here, so it must be identical across configurations; abl_kf plans
  against its own degraded field but is scored on the same one as everyone else.

Usage
-----
    python benchmark.py                 # 30 runs x 5 configs, all cores
    python benchmark.py --runs 5        # quick smoke test
    python benchmark.py --jobs 8
"""

import argparse
import importlib.util
import math
import os
import platform
import sys
import time
from collections import defaultdict

import numpy as np

# 11.py selects an interactive matplotlib backend at import time; force a
# headless one first so worker processes never try to open a window.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_algorithms():
    """Import 11.py by path (the name is not a valid Python identifier)."""
    path = os.path.join(HERE, "11.py")
    spec = importlib.util.spec_from_file_location("uav_algorithms", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["uav_algorithms"] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load_algorithms()

# Repair the three implementation defects documented in fixes.py before any
# simulation runs. 11.py is left untouched; the delta lives in fixes.py.
import fixes

fixes.apply_all(A)

import tuning

tuning.apply_all(A)

import dstar_lite

# Kalman-filter noise variances, swept by sensitivity.py. The nominal values
# (process 0.5, observation 2.0) reproduce the manuscript's setup; overriding
# these globals varies the filter's noise assumptions for the robustness study.
KF_PROCESS_NOISE = 0.5
KF_OBS_NOISE = 2.0

_orig_kf_init = A.KalmanFilter.__init__


def _kf_init_noise(self, dt, process_noise=0.5, measurement_noise=2.0):
    _orig_kf_init(self, dt, KF_PROCESS_NOISE, KF_OBS_NOISE)


A.KalmanFilter.__init__ = _kf_init_noise

# ---------------------------------------------------------------------------
# Constrained-kinematics UAV model (Review 4): add linear and angular
# acceleration limits a <= a_max and |omega_dot| <= omega_dot_max on top of the
# existing v <= v_max and |omega| <= omega_max. The stock UAV tracks commanded
# velocities instantaneously; the limits make the flown trajectory depend on the
# UAV's own dynamics, exposing whether the smoothed reference path is actually
# dynamically feasible.
# ---------------------------------------------------------------------------
UAV_MAX_ACCEL = 2.0        # m/s^2
UAV_MAX_ANG_ACCEL = 1.5    # rad/s^2

_orig_uav_init = A.UAV.__init__


def _uav_init(self, pos, radius=1.0, max_speed=2.0, max_turn_rate=0.8,
              detection_radius=30.0, max_accel=UAV_MAX_ACCEL,
              max_ang_accel=UAV_MAX_ANG_ACCEL):
    _orig_uav_init(self, pos, radius, max_speed, max_turn_rate, detection_radius)
    self.max_accel = max_accel
    self.max_ang_accel = max_ang_accel


A.UAV.__init__ = _uav_init


def _uav_update(self, v, omega, dt):
    v = max(0.0, v)
    dv = v - self.v
    dv = max(-self.max_accel * dt, min(self.max_accel * dt, dv))
    self.v += dv
    omega = max(-self.max_turn_rate, min(self.max_turn_rate, omega))
    dom = omega - self.omega
    dom = max(-self.max_ang_accel * dt, min(self.max_ang_accel * dt, dom))
    self.omega += dom
    self.heading += self.omega * dt
    self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
    self.pos[0] += self.v * math.cos(self.heading) * dt
    self.pos[1] += self.v * math.sin(self.heading) * dt
    dist = math.hypot(self.pos[0] - self.prev_pos[0], self.pos[1] - self.prev_pos[1])
    if dist < 0.01:
        self.stuck_timer += dt
    else:
        self.stuck_timer = 0
        self.prev_pos = self.pos.copy()


A.UAV.update = _uav_update

# --------------------------------------------------------------------------
# Nominal scenario, transcribed from 11.py main() lines 559-585.
# --------------------------------------------------------------------------
BOUNDS = (0, 200, 0, 200)
GRID_SIZE = (100, 100)
RESOLUTION = 2.0
START = (10.0, 10.0)
GOAL = (180.0, 180.0)

STATIC_OBSTACLES = [
    (40, 60, 8.0), (60, 40, 8.0), (70, 70, 6.0), (30, 30, 6.0),
    (50, 90, 10.0), (80, 50, 7.0), (100, 80, 5.0), (120, 100, 6.0),
    (140, 120, 5.0), (160, 60, 7.0), (110, 150, 8.0), (150, 180, 6.0),
]

# (x, y, radius, vx, vy) -- x/y are the nominal positions; only these are
# re-randomised per seed. radius/vx/vy are held fixed (paper Sec. 5.2).
DYNAMIC_OBSTACLES = [
    (20, 80, 4.0, 1.2, -0.8), (90, 30, 4.0, -0.8, 1.0),
    (50, 50, 3.0, 0.6, 0.8), (150, 150, 5.0, -1.0, 0.5),
    (80, 120, 3.5, 0.8, -1.2), (120, 30, 4.0, -0.5, 0.9),
    (180, 80, 4.5, -1.2, -0.6), (60, 180, 3.5, 1.0, -0.7),
    (100, 20, 4.0, 0.7, 1.1), (140, 60, 3.0, -0.9, -0.8),
]

UAV_KW = dict(radius=1.0, max_speed=2.5, max_turn_rate=0.9, detection_radius=35.0)

DT = 0.2
REPLAN_INTERVAL = 5.0
RISK_TRIGGER = 0.3
ASTAR_RISK_THRESHOLD = 0.6
BETA = 10.0  # sigmoid steepness of the risk-adaptive heuristic (Eq. 8)
SMOOTH_RISK_THRESHOLD = 0.7
GOAL_TOLERANCE = 5.0
MAX_SIM_TIME = 400.0
N_PRED_STEPS = 25

CONFIGS = {
    "astar":      dict(label="Traditional A*",        adaptive=False, astar_risk=False,
                       smooth=False, dwa=False, kf=False),
    "dstar_lite": dict(label="D* Lite",               adaptive=False, astar_risk=False,
                       smooth=True,  dwa=False, kf=False, planner="dstar"),
    "astar_dwa":  dict(label="A* + DWA",              adaptive=False, astar_risk=False,
                       smooth=True,  dwa=True,  kf=False, risk_switch=False),
    "astar_pred_dwa": dict(label="A* + Pred-DWA",     adaptive=False, astar_risk=False,
                       smooth=True,  dwa=True,  kf=False, risk_switch=False,
                       pred_dwa=True),
    "dwa_only":   dict(label="DWA-only",              adaptive=False, astar_risk=False,
                       smooth=False, dwa=True,  kf=False, planner="dwa_only",
                       risk_switch=False),
    "ia_only":    dict(label="IA*-only",              adaptive=True,  astar_risk=True,
                       smooth=True,  dwa=False, kf=True),
    "proposed":   dict(label="Proposed algorithm",    adaptive=True,  astar_risk=True,
                       smooth=True,  dwa=True,  kf=True),
    "proposed_pred_dwa": dict(label="Proposed + Pred-DWA", adaptive=True, astar_risk=True,
                       smooth=True,  dwa=True,  kf=True, pred_dwa=True),
    "abl_kf":     dict(label="Ablation 1 (w/o KF)",   adaptive=True,  astar_risk=True,
                       smooth=True,  dwa=True,  kf=False),
    "abl_adah":   dict(label="Ablation 2 (w/o AdaH)", adaptive=True,  astar_risk=True,
                       smooth=True,  dwa=True,  kf=True, euclid_only=True),
}
CONFIG_ORDER = ["astar", "dstar_lite", "astar_dwa", "astar_pred_dwa", "dwa_only",
                "ia_only", "proposed", "proposed_pred_dwa", "abl_kf", "abl_adah"]


# --------------------------------------------------------------------------
# Environment subclass: vectorised risk field + separate evaluation field
# --------------------------------------------------------------------------
class BenchEnvironment(A.Environment):
    """Environment with a numpy-vectorised risk field.

    Semantics are identical to Environment.compute_risk_grid in 11.py (per-cell
    max over obstacles and prediction steps, then max-normalisation); only the
    inner double loop is replaced by array arithmetic, for speed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_risk_grid = np.zeros(self.grid_size)

    def _risk_into(self, grid, uav, use_kf, n_steps=N_PRED_STEPS):
        grid.fill(0.0)
        # Global forward-looking cost map: build the field from ALL dynamic
        # obstacles (ground-truth states are available in simulation), so the
        # global planner can route around predicted future positions across the
        # whole map. The 35 m sensor radius is a *local* DWA concept and is
        # enforced separately in DynamicWindowApproach.calculate_velocity via
        # get_detected_obstacles. (Design decision D1 -- see the Tuning sheet.)
        res = self.resolution
        nx, ny = self.grid_size

        for obs in self.dynamic_obstacles:
            tracker = self.get_dynamic_tracker(obs)
            for step in range(1, n_steps + 1):
                if use_kf and tracker is not None and tracker.kf.initialized:
                    pred_state, pred_cov = tracker.get_predicted_state(step)
                    px, py = float(pred_state[0]), float(pred_state[1])
                    sigma = np.array(pred_cov[0:2, 0:2], dtype=float)
                else:
                    # constant-velocity extrapolation + fixed diagonal covariance
                    # (identical to the fallback branch of 11.py lines 238-242)
                    px = obs.x + obs.vx * step * DT
                    py = obs.y + obs.vy * step * DT
                    sigma = np.eye(2) * (1.0 + step * 0.5)

                sigma[0, 0] += obs.radius ** 2
                sigma[1, 1] += obs.radius ** 2

                gx, gy = int(px / res), int(py / res)
                spread = max(2, int(math.sqrt(sigma[0, 0] + sigma[1, 1]) / res) + 1)
                x0, x1 = max(0, gx - spread), min(nx, gx + spread + 1)
                y0, y1 = max(0, gy - spread), min(ny, gy + spread + 1)
                if x0 >= x1 or y0 >= y1:
                    continue

                xs = np.arange(x0, x1) * res - px
                ys = np.arange(y0, y1) * res - py
                X = xs[:, None]
                Y = ys[None, :]

                try:
                    si = np.linalg.inv(sigma)
                    maha = (si[0, 0] * X * X
                            + (si[0, 1] + si[1, 0]) * X * Y
                            + si[1, 1] * Y * Y)
                    risk = obs.risk_coeff * np.exp(-maha / 2.0)
                except np.linalg.LinAlgError:
                    sv2 = 2.0 * (sigma[0, 0] + sigma[1, 1]) + 1e-6
                    risk = obs.risk_coeff * np.exp(-(X * X + Y * Y) / sv2)

                block = grid[x0:x1, y0:y1]
                np.maximum(block, risk, out=block)

        # Each Gaussian is already <= 1 (exp(-.../2)), and we take the max over
        # obstacles/steps, so the field is already in [0,1]. Dividing by the
        # global max would re-inflate the peak to 1.0 regardless of the actual
        # risk and collapse the tail to ~0, which is what made the field (and
        # the DWA trigger) non-informative. Clamp only, per tuning.py T2.
        np.clip(grid, 0.0, 1.0, out=grid)
        return grid

    def compute_risk_grid(self, uav, current_time, time_window=3.0, n_steps=N_PRED_STEPS,
                          use_kf=True):
        """Planner-facing field (may be degraded for the w/o-KF ablation)."""
        return self._risk_into(self.risk_grid, uav, use_kf, n_steps)

    def compute_eval_risk_grid(self, uav, n_steps=N_PRED_STEPS):
        """Scoring field -- always Kalman-based, identical for all configs."""
        return self._risk_into(self.eval_risk_grid, uav, True, n_steps)

    def eval_risk_at(self, x, y):
        gx, gy = int(x / self.resolution), int(y / self.resolution)
        if 0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1]:
            return float(self.eval_risk_grid[gx, gy])
        return 0.0


class EuclideanAStar(A.AdaptiveAStar):
    """A* with the risk-adaptive heuristic disabled (pure Euclidean distance)."""

    def adaptive_heuristic(self, node, goal, risk_val):
        return self.heuristic(node, goal)


class RiskBlindAStar(EuclideanAStar):
    """Euclidean A* that also ignores the risk field when expanding nodes."""

    def get_neighbors(self, node):
        x, y = node
        out = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            nx_ = x + dx * self.step_size
            ny_ = y + dy * self.step_size
            if nx_ < self.env.bounds[0] or nx_ > self.env.bounds[1]:
                continue
            if ny_ < self.env.bounds[2] or ny_ > self.env.bounds[3]:
                continue
            if not self.env.is_collision(nx_, ny_, self.uav.radius, ignore_risk=True):
                out.append((round(nx_, 1), round(ny_, 1)))
        return out


# --------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------
class PredictionAwareDWA(A.DynamicWindowApproach):
    """DWA that advances dynamic obstacles in its forward simulation.

    The stock DWA checks the forward-simulated UAV trajectory against the
    *current* obstacle positions, i.e. it implicitly assumes obstacles are
    static over the 1.5 s prediction window. This variant instead propagates
    each dynamic obstacle using its Kalman-filter (or constant-velocity)
    prediction, so the local planner avoids where obstacles *will* be rather
    than where they are now.
    """

    def _predictive_obstacles(self):
        detected_static, detected_dynamic = self.env.get_detected_obstacles(self.uav)
        static = [(o.x, o.y, o.radius) for o in detected_static]
        horizon = max(1, int(self.predict_time / self.dt))
        dyn = []
        for obs in detected_dynamic:
            tracker = self.env.get_dynamic_tracker(obs)
            preds = []
            for k in range(1, horizon + 1):
                if tracker is not None and tracker.kf.initialized:
                    ps, _ = tracker.get_predicted_state(k)
                    preds.append((float(ps[0]), float(ps[1])))
                else:
                    preds.append((obs.x + obs.vx * k * self.dt,
                                  obs.y + obs.vy * k * self.dt))
            dyn.append((obs.radius, preds))
        return static, dyn

    def _simulate_predictive(self, v, omega, static, dyn):
        traj = [(self.uav.pos[0], self.uav.pos[1])]
        x, y = float(self.uav.pos[0]), float(self.uav.pos[1])
        heading = float(self.uav.heading)
        horizon = max(1, int(self.predict_time / self.dt))
        margin = fixes.SAFETY_MARGIN
        v_cur = self.uav.v
        w_cur = self.uav.omega
        a_max = getattr(self.uav, "max_accel", float("inf"))
        wa_max = getattr(self.uav, "max_ang_accel", float("inf"))
        for k in range(1, horizon + 1):
            v_cur += max(-a_max * self.dt, min(a_max * self.dt, v - v_cur))
            w_cur += max(-wa_max * self.dt, min(wa_max * self.dt, omega - w_cur))
            x += v_cur * math.cos(heading) * self.dt
            y += v_cur * math.sin(heading) * self.dt
            heading += w_cur * self.dt
            traj.append((x, y))
            for (ox, oy, r) in static:
                if math.hypot(x - ox, y - oy) < self.uav.radius + r + margin:
                    return None
            for (r, preds) in dyn:
                ox, oy = preds[k - 1]
                if math.hypot(x - ox, y - oy) < self.uav.radius + r + margin:
                    return None
        return traj

    def calculate_velocity(self, global_path, local_goal):
        static, dyn = self._predictive_obstacles()
        flat = static + [(preds[0][0], preds[0][1], r) for (r, preds) in dyn]

        goal_dir = (local_goal[0] - self.uav.pos[0], local_goal[1] - self.uav.pos[1])
        goal_dist = math.hypot(goal_dir[0], goal_dir[1])
        if goal_dist < 3.0:
            return self._go_towards(local_goal, self.uav.max_speed * 0.3)

        v_cruise = self.uav.max_speed * 0.8
        v_candidates = np.linspace(0.5, v_cruise, 12)
        omega_candidates = np.linspace(-self.uav.max_turn_rate, self.uav.max_turn_rate, 25)

        best_score = -float("inf")
        best_vel = (v_cruise, 0.0)
        has_valid = False
        for v in v_candidates:
            for omega in omega_candidates:
                traj = self._simulate_predictive(v, omega, static, dyn)
                if traj is None:
                    continue
                score = self._evaluate(traj, local_goal, global_path, flat)
                if score > best_score:
                    best_score = score
                    best_vel = (v, omega)
                    has_valid = True

        if not has_valid:
            best_vel = self._go_towards(local_goal, v_cruise * 0.6)
            test_traj = self._simulate_predictive(best_vel[0], best_vel[1], static, dyn)
            if test_traj is None:
                for omega in np.linspace(-self.uav.max_turn_rate, self.uav.max_turn_rate, 20):
                    tv = self._go_towards(local_goal, v_cruise * 0.3)
                    trial = (v_cruise * 0.3, max(-self.uav.max_turn_rate,
                             min(self.uav.max_turn_rate, tv[1] + omega)))
                    test_traj = self._simulate_predictive(trial[0], trial[1], static, dyn)
                    if test_traj is not None:
                        best_vel = trial
                        break
        return best_vel


# --------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------
class ManeuveringObstacle(A.Obstacle):
    """Dynamic obstacle with non-constant-velocity motion.

    Turns at a constant angular rate and changes speed at a constant
    acceleration, violating the constant-velocity assumption of the Kalman
    filter. This is used to test prediction robustness under model mismatch.
    """

    def __init__(self, x, y, radius, vx, vy, turn_rate=0.0, accel=0.0,
                 min_speed=0.3, max_speed=3.0):
        super().__init__(x, y, radius, risk_coeff=1.0, is_dynamic=True, vx=vx, vy=vy)
        self.turn_rate = turn_rate
        self.accel = accel
        self.min_speed = min_speed
        self.max_speed = max_speed

    def update(self, dt, bounds):
        speed = math.hypot(self.vx, self.vy)
        heading = math.atan2(self.vy, self.vx)
        heading += self.turn_rate * dt
        speed = max(self.min_speed, min(self.max_speed, speed + self.accel * dt))
        self.vx = speed * math.cos(heading)
        self.vy = speed * math.sin(heading)
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x < bounds[0] + self.radius:
            self.x = bounds[0] + self.radius
            self.vx = abs(self.vx)
        elif self.x > bounds[1] - self.radius:
            self.x = bounds[1] - self.radius
            self.vx = -abs(self.vx)
        if self.y < bounds[2] + self.radius:
            self.y = bounds[2] + self.radius
            self.vy = abs(self.vy)
        elif self.y > bounds[3] - self.radius:
            self.y = bounds[3] - self.radius
            self.vy = -abs(self.vy)


def build_scenario(seed, maneuver=False):
    """Same seed -> identical environment for every configuration."""
    rng = np.random.default_rng(seed)
    env = BenchEnvironment(grid_size=GRID_SIZE, resolution=RESOLUTION, bounds=BOUNDS)

    for (x, y, r) in STATIC_OBSTACLES:
        env.add_static_obstacle(x, y, r)

    for (_nx, _ny, radius, vx, vy) in DYNAMIC_OBSTACLES:
        # Randomise the initial position only; retain radius and velocity.
        for _ in range(200):
            x = float(rng.uniform(BOUNDS[0] + radius, BOUNDS[1] - radius))
            y = float(rng.uniform(BOUNDS[2] + radius, BOUNDS[3] - radius))
            # keep clear of static obstacles and of the start/goal cells
            if any(math.hypot(x - sx, y - sy) < r + radius + 2.0
                   for (sx, sy, r) in STATIC_OBSTACLES):
                continue
            if math.hypot(x - START[0], y - START[1]) < 25.0:
                continue
            if math.hypot(x - GOAL[0], y - GOAL[1]) < 25.0:
                continue
            break
        if maneuver:
            turn_rate = float(rng.uniform(-0.6, 0.6))
            accel = float(rng.uniform(-0.4, 0.4))
            obs = ManeuveringObstacle(x, y, radius, vx, vy, turn_rate, accel)
            env.dynamic_obstacles.append(obs)
            tracker = A.ObstacleTracker(len(env.obstacle_trackers), dt=DT)
            tracker.kf.initialize(x, y, 0.0, 0.0)  # zero-velocity prior (FIX 5)
            env.obstacle_trackers.append(tracker)
        else:
            env.add_dynamic_obstacle(x, y, radius, vx, vy)

    uav = A.UAV(pos=START, **UAV_KW)
    uav.heading = math.atan2(GOAL[1] - START[1], GOAL[0] - START[0])
    return env, uav


def in_collision(env, uav):
    """Paper Sec. 5.2 criterion, evaluated against every obstacle in the scene."""
    for obs in env.static_obstacles:
        if math.hypot(uav.pos[0] - obs.x, uav.pos[1] - obs.y) < uav.radius + obs.radius + 0.5:
            return True
    for obs in env.dynamic_obstacles:
        if math.hypot(uav.pos[0] - obs.x, uav.pos[1] - obs.y) < uav.radius + obs.radius + 0.5:
            return True
    return False


def mean_risk_around(env, uav, radius=5.0):
    total, count = 0.0, 0
    for dx in np.linspace(-radius, radius, 7):
        for dy in np.linspace(-radius, radius, 7):
            total += env.eval_risk_at(uav.pos[0] + dx, uav.pos[1] + dy)
            count += 1
    return total / count if count else 0.0


def risk_at_uav(env, uav):
    """Risk-field value at the UAV's own grid cell (the metric of Section 5.2:
    the average of the risk field at sampling points along the trajectory)."""
    return env.eval_risk_at(uav.pos[0], uav.pos[1])


def planner_risk_box(env, uav, radius=5.0):
    """Local risk for the algorithm's own control flow (DWA trigger / recovery),
    sampled from the *planner's* field (use_kf per config), not the scoring
    field. The w/o-KF ablation must trigger on its constant-velocity field."""
    total, count = 0.0, 0
    for dx in np.linspace(-radius, radius, 7):
        for dy in np.linspace(-radius, radius, 7):
            total += env.get_risk_at(uav.pos[0] + dx, uav.pos[1] + dy)
            count += 1
    return total / count if count else 0.0


# --------------------------------------------------------------------------
# Static-only smoothing (for the risk-blind D* Lite baseline)
# --------------------------------------------------------------------------
def static_line_safe(env, p1, p2, risk_threshold=0.6, robot_radius=1.0):
    """Line-of-sight against STATIC obstacles only (D* Lite is risk-blind)."""
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if dist < 1e-6:
        return True
    steps = max(int(dist / env.resolution), 5)
    for i in range(steps + 1):
        t = i / steps
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        for obs in env.static_obstacles:
            if math.hypot(x - obs.x, y - obs.y) < robot_radius + obs.radius + 0.5:
                return False
    return True


def smooth_static_only(path, env, robot_radius, num_points=150):
    """Smooth a grid path using static-only collision checks (for D* Lite)."""
    for cap in fixes.SHORTCUT_CAPS:
        reduced = fixes.remove_redundant_capped(path, env, static_line_safe, 0.6, cap)
        curve = A.PathSmoother.cubic_spline_smooth(reduced, num_points=num_points)
        if fixes.spline_is_clear(curve, env, robot_radius):
            return curve
    return path


# --------------------------------------------------------------------------
# Episode
# --------------------------------------------------------------------------
def run_episode(cfg_name, seed, pred_steps=N_PRED_STEPS, maneuver=False):
    cfg = CONFIGS[cfg_name]
    use_kf = cfg["kf"]
    use_dwa = cfg["dwa"]
    use_smooth = cfg["smooth"]
    risk_switch = cfg.get("risk_switch", True)
    euclid_only = cfg.get("euclid_only", False) or not cfg["adaptive"]
    astar_risk = cfg["astar_risk"] and not cfg.get("euclid_only", False)
    planner_kind = cfg.get("planner", "astar")

    env, uav = build_scenario(seed, maneuver)
    smoother = A.PathSmoother()
    if cfg.get("pred_dwa", False):
        dwa = PredictionAwareDWA(uav, env, GOAL)
    else:
        dwa = A.DynamicWindowApproach(uav, env, GOAL)

    t_wall0 = time.perf_counter()

    dstar = None
    if planner_kind == "dstar":
        dstar = dstar_lite.DStarLite(
            GRID_SIZE, RESOLUTION, BOUNDS, STATIC_OBSTACLES,
            [(o.x, o.y, o.radius) for o in env.dynamic_obstacles],
            (uav.pos[0], uav.pos[1]), GOAL,
            robot_radius=uav.radius, margin=0.5)
        dstar.update_dynamic_obstacles(
            [(o.x, o.y, o.radius) for o in env.dynamic_obstacles])

    if not cfg["adaptive"] or cfg.get("euclid_only", False):
        planner_cls = RiskBlindAStar if not astar_risk else EuclideanAStar
    else:
        planner_cls = A.AdaptiveAStar

    replan_times = []
    spline_fallbacks = [0]

    def do_replan():
        t0 = time.perf_counter()
        # The Kalman trackers underpin the *scoring* field, which must be
        # identical for every configuration. Update unconditionally.
        env.update_kf_trackers(sim_time)
        env.compute_risk_grid(uav, sim_time, use_kf=use_kf, n_steps=pred_steps)
        if planner_kind == "dstar":
            raw = dstar.plan((uav.pos[0], uav.pos[1]))
            out = smooth_static_only(raw, env, uav.radius) if use_smooth else raw
        else:
            planner = planner_cls(env, uav, GOAL, risk_threshold=ASTAR_RISK_THRESHOLD, k=BETA)
            raw = planner.run()
            if use_smooth:
                out, _n_ctrl, cap = fixes.safe_smooth(
                    raw, env, uav.radius, A.PathSmoother,
                    risk_threshold=SMOOTH_RISK_THRESHOLD, num_points=150)
                if cap is None:
                    spline_fallbacks[0] += 1
            else:
                out = raw
        replan_times.append(time.perf_counter() - t0)
        return out

    sim_time = 0.0
    if planner_kind == "dwa_only":
        path = None
    else:
        env.compute_risk_grid(uav, 0.0, use_kf=use_kf, n_steps=pred_steps)
        path = do_replan()

    last_replan = 0.0
    dwa_active = False
    dwa_start = 0.0
    consecutive_high = 0
    n_dwa_activations = 0
    n_dwa_steps = 0

    traj_len = 0.0
    risk_samples = []
    peak_omega = 0.0
    omega_utils = []
    max_risk = 0.0
    tracking_errors = []
    prev = (uav.pos[0], uav.pos[1])

    outcome = "timeout"
    n_steps = int(MAX_SIM_TIME / DT)

    for _ in range(n_steps):
        sim_time += DT
        env.update_dynamic(DT)
        # Update the Kalman trackers unconditionally so the scoring field
        # (compute_eval_risk_grid) is a live Kalman field for every config,
        # not a stale field frozen at the obstacle's initial state.
        env.update_kf_trackers(sim_time)
        env.compute_risk_grid(uav, sim_time, use_kf=use_kf, n_steps=pred_steps)
        env.compute_eval_risk_grid(uav, n_steps=pred_steps)

        if planner_kind == "dstar":
            dstar.update_dynamic_obstacles(
                [(o.x, o.y, o.radius) for o in env.dynamic_obstacles])
            path = do_replan()

        if in_collision(env, uav):
            outcome = "collision"
            break
        if uav.distance_to(GOAL) < GOAL_TOLERANCE:
            outcome = "success"
            break

        replan_trigger = (sim_time - last_replan) >= REPLAN_INTERVAL
        if use_dwa and dwa_active and (sim_time - dwa_start > 3.0):
            if planner_risk_box(env, uav) < 0.15:
                dwa_active = False
                replan_trigger = True
        if replan_trigger and planner_kind != "dwa_only":
            path = do_replan()
            last_replan = sim_time

        local_risk = planner_risk_box(env, uav)
        risk_samples.append(risk_at_uav(env, uav))
        max_risk = max(max_risk, risk_at_uav(env, uav))

        if use_dwa and risk_switch:
            if local_risk > RISK_TRIGGER:
                consecutive_high += 1
            else:
                consecutive_high = 0
            if consecutive_high >= 2 and not dwa_active:
                dwa_active = True
                dwa_start = sim_time
                consecutive_high = 0
                n_dwa_activations += 1
            if dwa_active:
                n_dwa_steps += 1

        if planner_kind == "dwa_only":
            v, omega = dwa.calculate_velocity(None, GOAL)
        elif use_dwa:
            if dwa_active:
                target = find_lookahead(path, uav.pos, 3.0, 10)
                v, omega = dwa.calculate_velocity(path, target)
            else:
                target = find_lookahead(path, uav.pos, 8.0, 25)
                v, omega = dwa.calculate_velocity(path, target)
        else:
            # pure pursuit along the reference path
            target = find_lookahead(path, uav.pos, 8.0, 25)
            v, omega = dwa._go_towards(target, uav.max_speed * 0.8)

        uav.update(v, omega, DT)

        if path is not None and len(path) > 0:
            tracking_errors.append(min(
                math.hypot(uav.pos[0] - p[0], uav.pos[1] - p[1]) for p in path))

        peak_omega = max(peak_omega, abs(uav.omega))
        omega_utils.append(100.0 * abs(uav.omega) / uav.max_turn_rate)
        cur = (uav.pos[0], uav.pos[1])
        traj_len += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        prev = cur

        if uav.stuck_timer > 20.0:
            outcome = "stuck"
            break

    success = outcome == "success"
    wall_time = time.perf_counter() - t_wall0
    full_step_time = wall_time / max(1, int(round(sim_time / DT)))
    return dict(
        config=cfg_name,
        seed=seed,
        success=int(success),
        outcome=outcome,
        path_length=traj_len if success else float("nan"),
        mean_risk=(float(np.mean(risk_samples)) if (success and risk_samples) else float("nan")),
        max_risk=(float(max_risk) if (success and risk_samples) else float("nan")),
        peak_omega_util=(100.0 * peak_omega / uav.max_turn_rate) if success else float("nan"),
        mean_omega_util=(float(np.mean(omega_utils)) if (success and omega_utils) else float("nan")),
        p95_omega_util=(float(np.percentile(omega_utils, 95)) if (success and omega_utils) else float("nan")),
        tracking_error=(float(np.mean(tracking_errors)) if (success and tracking_errors) else float("nan")),
        replan_time=float(np.mean(replan_times)) if replan_times else float("nan"),
        n_replans=len(replan_times),
        sim_time=sim_time,
        n_dwa_activations=n_dwa_activations,
        dwa_duty=(100.0 * n_dwa_steps / max(1, int(sim_time / DT))) if use_dwa else 0.0,
        spline_fallbacks=spline_fallbacks[0],
        wall_time=wall_time,
        full_step_time=full_step_time,
    )


def find_lookahead(path, pos, min_dist, max_offset):
    if not path:
        return GOAL
    best_i, best_d = 0, float("inf")
    for i, p in enumerate(path):
        d = math.hypot(pos[0] - p[0], pos[1] - p[1])
        if d < best_d:
            best_d, best_i = d, i
    for s in range(1, max_offset + 1):
        idx = min(best_i + s, len(path) - 1)
        if math.hypot(pos[0] - path[idx][0], pos[1] - path[idx][1]) >= min_dist:
            return path[idx]
    return path[-1]


def _worker(job):
    cfg_name, seed, pred_steps, maneuver = job
    try:
        return run_episode(cfg_name, seed, pred_steps, maneuver)
    except Exception as exc:  # keep one bad episode from killing the sweep
        return dict(config=cfg_name, seed=seed, success=0, outcome="error:%s" % exc,
                    path_length=float("nan"), mean_risk=float("nan"),
                    max_risk=float("nan"), peak_omega_util=float("nan"),
                    mean_omega_util=float("nan"), p95_omega_util=float("nan"),
                    tracking_error=float("nan"),
                    replan_time=float("nan"), n_replans=0, sim_time=0.0,
                    wall_time=0.0, full_step_time=float("nan"))


# --------------------------------------------------------------------------
# Aggregation and output
# --------------------------------------------------------------------------
def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion (in [0, 1])."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarise(rows):
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)

    summary = []
    for name in CONFIG_ORDER:
        rs = by_cfg[name]
        n = len(rs)
        succ = [r for r in rs if r["success"]]
        def m(key, src):
            vals = [r[key] for r in src if not math.isnan(r[key])]
            return float(np.mean(vals)) if vals else float("nan")
        def sd(key, src):
            vals = [r[key] for r in src if not math.isnan(r[key])]
            return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        _lo, _hi = wilson_ci(len(succ), n)
        summary.append(dict(
            config=name,
            label=CONFIGS[name]["label"],
            n_runs=n,
            n_success=len(succ),
            success_rate=100.0 * len(succ) / n if n else float("nan"),
            success_lo=100.0 * _lo,                  success_hi=100.0 * _hi,
            path_length=m("path_length", succ),      path_length_sd=sd("path_length", succ),
            mean_risk=m("mean_risk", succ),          mean_risk_sd=sd("mean_risk", succ),
            max_risk=m("max_risk", succ),
            peak_omega=m("peak_omega_util", succ),   peak_omega_sd=sd("peak_omega_util", succ),
            mean_omega=m("mean_omega_util", succ),   mean_omega_sd=sd("mean_omega_util", succ),
            p95_omega=m("p95_omega_util", succ),     p95_omega_sd=sd("p95_omega_util", succ),
            tracking_error=m("tracking_error", succ),
            replan_time=m("replan_time", rs),        replan_time_sd=sd("replan_time", rs),
            full_step_time=m("full_step_time", rs),
            n_collision=sum(1 for r in rs if r["outcome"] == "collision"),
            n_timeout=sum(1 for r in rs if r["outcome"] == "timeout"),
            n_stuck=sum(1 for r in rs if r["outcome"] == "stuck"),
            dwa_activations=float(np.mean([r["n_dwa_activations"] for r in rs])) if rs else 0.0,
            dwa_duty=float(np.mean([r["dwa_duty"] for r in rs])) if rs else 0.0,
            spline_fallbacks=sum(r["spline_fallbacks"] for r in rs),
        ))
    return summary


def pairwise_tests(rows, ref="proposed"):
    """Paired comparisons of `ref` against every other configuration.

    * success/failure over paired seeds: exact McNemar test (binomtest).
    * continuous metrics (path length, mean risk) over common-successful seeds:
      paired t-test and Wilcoxon signed-rank test.
    """
    from scipy import stats as st

    piv = defaultdict(dict)
    for r in rows:
        piv[r["config"]][r["seed"]] = r
    if ref not in piv:
        return {}

    out = {}
    for cfg in CONFIG_ORDER:
        if cfg == ref:
            continue
        seeds = sorted(set(piv[ref]) & set(piv[cfg]))
        if not seeds:
            continue
        b = c = 0  # b: ref succeeds & cfg fails;  c: ref fails & cfg succeeds
        for s in seeds:
            rs = piv[ref][s]["success"]
            cs = piv[cfg][s]["success"]
            if rs and not cs:
                b += 1
            elif not rs and cs:
                c += 1
        mcnemar_p = float("nan")
        if b + c > 0:
            mcnemar_p = st.binomtest(min(b, c), b + c, 0.5).pvalue
        common = [s for s in seeds
                  if piv[ref][s]["success"] and piv[cfg][s]["success"]]
        res = dict(b=b, c=c, mcnemar_p=mcnemar_p,
                   n_paired=len(seeds), n_common_success=len(common))
        for key in ("path_length", "mean_risk"):
            a_vals = [piv[ref][s][key] for s in common
                      if not math.isnan(piv[ref][s][key])]
            b_vals = [piv[cfg][s][key] for s in common
                      if not math.isnan(piv[cfg][s][key])]
            n = min(len(a_vals), len(b_vals))
            if n >= 2:
                a_vals, b_vals = a_vals[:n], b_vals[:n]
                _t, tp = st.ttest_rel(a_vals, b_vals)
                try:
                    _w, wp = st.wilcoxon(a_vals, b_vals)
                except ValueError:
                    wp = float("nan")
                res[key + "_t_p"] = tp
                res[key + "_wilcox_p"] = wp
                res[key + "_diff"] = float(np.mean(b_vals) - np.mean(a_vals))
            else:
                res[key + "_t_p"] = float("nan")
                res[key + "_wilcox_p"] = float("nan")
                res[key + "_diff"] = float("nan")
        out[cfg] = res
    return out


def write_excel(rows, summary, tests, path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment

    wb = Workbook()
    bold = Font(bold=True)

    ws = wb.active
    ws.title = "Summary"
    hdr = ["Configuration", "Runs", "Successes", "Success Rate (%)",
           "95% CI Low", "95% CI High",
           "Path Length (m)", "SD", "Mean Local Risk", "SD", "Max Local Risk",
           "Peak Ang. Vel. Util. (%)", "SD", "Mean Ang. Vel. Util. (%)", "SD",
           "P95 Ang. Vel. Util. (%)", "SD", "Tracking Error (m)",
           "Replanning Time (s)", "SD", "Full Step Time (s)",
           "Collisions", "Timeouts", "Stuck",
           "Mean DWA Activations", "DWA Duty (%)", "Spline Fallbacks"]
    ws.append(hdr)
    for c in ws[1]:
        c.font = bold
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for s in summary:
        ws.append([s["label"], s["n_runs"], s["n_success"], round(s["success_rate"], 1),
                   round(s["success_lo"], 1), round(s["success_hi"], 1),
                   round(s["path_length"], 1), round(s["path_length_sd"], 1),
                   round(s["mean_risk"], 4), round(s["mean_risk_sd"], 4),
                   round(s["max_risk"], 3),
                   round(s["peak_omega"], 1), round(s["peak_omega_sd"], 1),
                   round(s["mean_omega"], 1), round(s["mean_omega_sd"], 1),
                   round(s["p95_omega"], 1), round(s["p95_omega_sd"], 1),
                   round(s["tracking_error"], 2),
                   round(s["replan_time"], 3), round(s["replan_time_sd"], 3),
                   round(s["full_step_time"], 4),
                   s["n_collision"], s["n_timeout"], s["n_stuck"],
                   round(s["dwa_activations"], 2), round(s["dwa_duty"], 1),
                   s["spline_fallbacks"]])
    ws.column_dimensions["A"].width = 24
    for col in "BCDEFGHIJKLMNOPQRSTUVW":
        ws.column_dimensions[col].width = 11

    ws2 = wb.create_sheet("Raw runs")
    hdr2 = ["Configuration", "Seed", "Success", "Outcome", "Path Length (m)",
            "Mean Local Risk", "Max Local Risk", "Peak Ang. Vel. Util. (%)",
            "Mean Ang. Vel. Util. (%)", "P95 Ang. Vel. Util. (%)",
            "Tracking Error (m)", "Mean Replanning Time (s)", "Replans", "Sim Time (s)",
            "DWA Activations", "DWA Duty (%)", "Spline Fallbacks"]
    ws2.append(hdr2)
    for c in ws2[1]:
        c.font = bold
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    order = {n: i for i, n in enumerate(CONFIG_ORDER)}
    for r in sorted(rows, key=lambda r: (order[r["config"]], r["seed"])):
        def nz(v, nd):
            return "" if (isinstance(v, float) and math.isnan(v)) else round(v, nd)
        ws2.append([CONFIGS[r["config"]]["label"], r["seed"], r["success"], r["outcome"],
                    nz(r["path_length"], 2), nz(r["mean_risk"], 4), nz(r["max_risk"], 3),
                    nz(r["peak_omega_util"], 2), nz(r["mean_omega_util"], 2),
                    nz(r["p95_omega_util"], 2), nz(r["tracking_error"], 2), nz(r["replan_time"], 4),
                    r["n_replans"], round(r["sim_time"], 1),
                    r["n_dwa_activations"], round(r["dwa_duty"], 1),
                    r["spline_fallbacks"]])
    ws2.column_dimensions["A"].width = 24
    ws2.column_dimensions["D"].width = 14
    for col in "BCEFGHIJ":
        ws2.column_dimensions[col].width = 13
    ws2.freeze_panes = "A2"

    ws3 = wb.create_sheet("Setup")
    for k, v in [
        ("Map", "%d x %d m" % (BOUNDS[1], BOUNDS[3])),
        ("Grid resolution", "%.1f m" % RESOLUTION),
        ("Grid size", "%d x %d" % GRID_SIZE),
        ("Static obstacles", len(STATIC_OBSTACLES)),
        ("Dynamic obstacles", len(DYNAMIC_OBSTACLES)),
        ("Start", str(START)), ("Goal", str(GOAL)),
        ("Goal tolerance", "%.1f m" % GOAL_TOLERANCE),
        ("Max linear velocity", "%.1f m/s" % UAV_KW["max_speed"]),
        ("Max angular velocity", "%.1f rad/s" % UAV_KW["max_turn_rate"]),
        ("Detection radius", "%.1f m" % UAV_KW["detection_radius"]),
        ("UAV radius", "%.1f m" % UAV_KW["radius"]),
        ("A* step size", "4.0 m"),
        ("A* risk threshold", ASTAR_RISK_THRESHOLD),
        ("DWA risk trigger", RISK_TRIGGER),
        ("DWA prediction time", "1.5 s"),
        ("DWA velocity samples", "12 linear x 25 angular"),
        ("Replanning interval", "%.1f s" % REPLAN_INTERVAL),
        ("Simulation step", "%.1f s" % DT),
        ("Prediction steps", N_PRED_STEPS),
        ("Process noise variance", 0.5),
        ("Observation noise variance", 2.0),
        ("Max sim time per run", "%.0f s" % MAX_SIM_TIME),
        ("Randomisation", "dynamic obstacle initial positions only; radii and velocities fixed"),
        ("Seeding", "paired -- seed k gives an identical scene to every configuration"),
        ("Trajectory metrics", "aggregated over successful runs only"),
        ("Replanning time", "aggregated over all runs; excludes DWA and plotting"),
        ("Risk scoring field", "Kalman-based for every configuration"),
        ("Mean local risk", "risk field value at the UAV's own grid cell, averaged over the trajectory (Section 5.2 definition)"),
        ("Angular velocity metric", "peak is reported but is degenerate (every run hits 100% on the first heading step); mean and P95 utilisation are the meaningful smoothness metrics"),
        ("Code base", "11.py + fixes.py (3 bug fixes) + tuning.py (spec alignment); see Tuning sheet"),
        ("New baselines", "D* Lite (incremental replan every step), plain A*+DWA, DWA-only"),
        ("Hardware OS", platform.platform()),
        ("CPU cores", os.cpu_count() or 1),
        ("Python", platform.python_version()),
    ]:
        ws3.append([k, v])
    for c in ws3["A"]:
        c.font = bold
    ws3.column_dimensions["A"].width = 30
    ws3.column_dimensions["B"].width = 62

    ws4 = wb.create_sheet("Corrections")
    ws4.append(["#", "Location in 11.py", "Defect", "Correction"])
    for c in ws4[1]:
        c.font = bold
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for row in [
        [1, "line 536, DynamicWindowApproach._evaluate",
         "min_obs_dist seeded at 0.0 and only ever reduced, so it stays 0.0 for any "
         "trajectory with positive clearance. obs_score is then 1.0 for every "
         "candidate, making the sigma_2 obstacle term of Eq. (11) an additive "
         "constant with no effect on the argmax.",
         "Seed min_obs_dist at +inf so the term measures true minimum clearance."],
        [2, "line 521, _simulate_with_collision_check",
         "Trajectories were rejected at radius+obs_r+0.3, while Section 5.2 scores a "
         "collision at radius+obs_r+0.5. DWA treated a 0.2 m band as flyable that the "
         "metric already counted as a crash.",
         "Use the same 0.5 m margin as the collision criterion."],
        [3, "lines 407-439, PathSmoother",
         "remove_redundant validates shortcuts by line-of-sight, then "
         "cubic_spline_smooth replaces those segments with an unverified curve. The "
         "greedy shortcut collapsed ~48 waypoints to 3-5, so the spline bowed off the "
         "validated polyline (measured: +1.94 m clearance on the polyline, -2.56 m on "
         "the spline), passing through static obstacles.",
         "Cap the shortcut length so enough control points survive, verify the "
         "interpolated curve, retry with tighter caps, and fall back to the "
         "LOS-validated polyline if no cap yields a clear curve."],
    ]:
        ws4.append(row)
    ws4.column_dimensions["A"].width = 4
    ws4.column_dimensions["B"].width = 38
    ws4.column_dimensions["C"].width = 70
    ws4.column_dimensions["D"].width = 60
    for r in ws4.iter_rows(min_row=2):
        for c in r:
            c.alignment = Alignment(wrap_text=True, vertical="top")

    ws5 = wb.create_sheet("Tuning")
    ws5.append(["ID", "Kind", "What changed", "Why (paper text it aligns to)"])
    for c in ws5[1]:
        c.font = bold
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for row in [
        ["T1", "Spec alignment",
         "Risk-adaptive heuristic rewritten to implement Eq. (7)-(10) verbatim: "
         "f = g + (1-alpha)h_dist + alpha*h_risk, alpha = 1/(1+exp(-beta(rbar-r0))), "
         "h_risk = rbar*d_max.",
         "11.py used a non-convex double weight (alpha in [0.5,2]) with h_risk = "
         "rbar*h_dist, contradicting the manuscript and breaking admissibility."],
        ["T2", "Spec alignment",
         "Removed the redundant divide-by-global-max step in the risk-field "
         "normalization (clamp only).",
         "Each Gaussian is already <=1 so the field is in [0,1]; dividing by the max "
         "re-inflated the peak to 1.0 and collapsed the tail, making the field and "
         "DWA trigger non-informative."],
        ["D1", "Design decision",
         "Global planner risk field built from ALL dynamic obstacles (ground-truth "
         "states), not only the 35 m sensor footprint.",
         "Abstract/Sec 3.4 describe a forward-looking cost map; 11.py built it only "
         "within 35 m, giving no foresight beyond. 35 m radius retained for the "
         "local DWA layer."],
        ["D2", "Design decision",
         "Prediction horizon = 25 steps (5.0 s), matching the 5 s replanning period.",
         "11.py used 5 steps (1.0 s), far shorter than the replan interval, so the "
         "field was not forward-looking."],
        ["H1", "Measurement",
         "Mean local risk sampled at the UAV own grid cell (Sec 5.2 definition), "
         "over successful runs.",
         "The +/-5 m box mean gave ~0 because the field is localized; the box mean is "
         "still used for the DWA trigger (a separate quantity)."],
        ["H2", "Measurement",
         "Angular-velocity utilisation reported as P95 and mean, not peak.",
         "Peak is degenerate in this code: every run saturates the turn-rate on the "
         "first heading step, so peak is always 100%."],
        ["F1", "Bug fix",
         "DWA _evaluate: min_obs_dist seeded at +inf (was 0.0), so the obstacle term "
         "measures true clearance.",
         "Seeding at 0.0 made obs_score a constant 1.0 for every candidate (Eq. 11 "
         "term inert)."],
        ["F2", "Bug fix",
         "DWA collision margin 0.3 -> 0.5, matching the Sec 5.2 collision criterion.",
         "DWA treated a 0.3-0.5 m band as flyable that the metric already scored as a "
         "crash."],
        ["F3", "Bug fix",
         "Spline re-validated; shortcut length capped; LOS-validated polyline "
         "fallback.",
         "remove_redundant collapsed ~48 waypoints to 3-5 and the spline bowed off the "
         "validated segments (clearance +1.94 m polyline vs -2.56 m spline)."],
    ]:
        ws5.append(row)
    ws5.column_dimensions["A"].width = 4
    ws5.column_dimensions["B"].width = 16
    ws5.column_dimensions["C"].width = 72
    ws5.column_dimensions["D"].width = 72
    for r in ws5.iter_rows(min_row=2):
        for c in r:
            c.alignment = Alignment(wrap_text=True, vertical="top")

    def _num(v, nd):
        return "" if (isinstance(v, float) and math.isnan(v)) else round(v, nd)

    ws6 = wb.create_sheet("Statistical tests")
    ws6.append(["Baseline", "Paired seeds", "McNemar b", "McNemar c", "McNemar p",
                "Common success", "path diff (base-proposed)", "path t p", "path Wilcox p",
                "risk diff (base-proposed)", "risk t p", "risk Wilcox p"])
    for c in ws6[1]:
        c.font = bold
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for cfg, t in tests.items():
        ws6.append([CONFIGS[cfg]["label"], t["n_paired"], t["b"], t["c"],
                    _num(t["mcnemar_p"], 4), t["n_common_success"],
                    _num(t["path_length_diff"], 3), _num(t["path_length_t_p"], 4),
                    _num(t["path_length_wilcox_p"], 4),
                    _num(t["mean_risk_diff"], 4), _num(t["mean_risk_t_p"], 4),
                    _num(t["mean_risk_wilcox_p"], 4)])
    ws6.column_dimensions["A"].width = 22
    for col in "BCDEFGHIJKLMNOP":
        ws6.column_dimensions[col].width = 13

    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default="experiment_data.xlsx")
    ap.add_argument("--pred-steps", type=int, default=N_PRED_STEPS)
    ap.add_argument("--configs", default=None,
                    help="comma-separated config names to run (default: all)")
    ap.add_argument("--maneuver", action="store_true",
                    help="use maneuvering (turning/accelerating) dynamic obstacles")
    args = ap.parse_args()

    if args.configs:
        selected = [c.strip() for c in args.configs.split(",") if c.strip()]
        order = [c for c in CONFIG_ORDER if c in selected]
    else:
        order = CONFIG_ORDER

    jobs = [(c, s, args.pred_steps, args.maneuver) for c in order for s in range(args.runs)]
    print("Running %d episodes (%d configs x %d seeds) on %d workers ...\n"
          % (len(jobs), len(order), args.runs, args.jobs))

    t0 = time.time()
    rows = []
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, jobs), 1):
                rows.append(r)
                print("  [%3d/%3d] %-22s seed %-3d %-9s  %6.1f s"
                      % (i, len(jobs), CONFIGS[r["config"]]["label"], r["seed"],
                         r["outcome"], time.time() - t0), flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            r = _worker(j)
            rows.append(r)
            print("  [%3d/%3d] %-22s seed %-3d %-9s" % (i, len(jobs),
                  CONFIGS[r["config"]]["label"], r["seed"], r["outcome"]), flush=True)

    summary = summarise(rows)
    tests = pairwise_tests(rows, ref="proposed")

    print("\n" + "=" * 160)
    print("%-22s %6s %9s %-16s %11s %11s %10s %10s %12s %9s %12s" % (
        "Configuration", "Runs", "Success%", "95% CI", "Path (m)",
        "Mean risk", "Max risk", "TrackErr", "Full step(s)", "Replan(s)", "Fail(C/T/S)"))
    print("-" * 160)
    for s in summary:
        ci = "[%4.1f,%4.1f]" % (s["success_lo"], s["success_hi"])
        fail = "%d/%d/%d" % (s["n_collision"], s["n_timeout"], s["n_stuck"])
        print("%-22s %6d %9.1f %-16s %11.1f %11.4f %10.3f %10.2f %12.4f %9.3f %12s" % (
            s["label"], s["n_runs"], s["success_rate"], ci, s["path_length"],
            s["mean_risk"], s["max_risk"], s["tracking_error"], s["full_step_time"],
            s["replan_time"], fail))
    print("=" * 160)
    print("Path length / risk / angular velocity averaged over successful runs only;")
    print("replanning time and full step time over all runs. Fail = collisions/timeouts/stuck.")
    print("Success-rate 95% CI is the Wilson score interval.")

    if tests:
        n_paired = max(t["n_paired"] for t in tests.values())
        print("\nPaired statistical tests vs Proposed algorithm (%d paired seeds):" % n_paired)
        print("%-22s %-9s %-8s %-10s %-10s %-10s %-10s" % (
            "Baseline", "McNemar p", "b/c", "path t p", "path W p", "risk t p", "risk W p"))
        for cfg, t in tests.items():
            print("%-22s %-9.3f %-8s %-10.3f %-10.3f %-10.3f %-10.3f" % (
                CONFIGS[cfg]["label"], t["mcnemar_p"], "%d/%d" % (t["b"], t["c"]),
                t["path_length_t_p"], t["path_length_wilcox_p"],
                t["mean_risk_t_p"], t["mean_risk_wilcox_p"]))

    print("\nHardware / platform:")
    print("  OS: %s | CPU: %d cores | Python %s" % (
        platform.platform(), os.cpu_count() or 1, platform.python_version()))

    out = os.path.join(HERE, args.out)
    write_excel(rows, summary, tests, out)
    print("\nWrote %s" % out)
    print("Total wall time: %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
