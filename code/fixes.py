"""
fixes.py -- Corrections to three implementation defects in 11.py.

Each patch below repairs a defect in which the *code* contradicts the behaviour
the manuscript already describes. None of them changes the algorithm's design,
its parameters, or any reported quantity by construction; they make the
implementation do what Sections 4.2 and 4.3 say it does.

Applied by calling apply_all() before running a simulation. 11.py itself is left
untouched, so the delta stays auditable.

--------------------------------------------------------------------------
FIX 1 -- DWA obstacle-distance term was inert            (11.py line 536)
--------------------------------------------------------------------------
    min_obs_dist = 0.0
    ...
    min_obs_dist = min(min_obs_dist, d)

`min_obs_dist` is seeded at 0.0 and then only ever reduced, so for any trajectory
whose clearances are all positive it stays exactly 0.0. That yields
`obs_score = 1.0 - 0.0/3.0 = 1.0` for *every* candidate, making the sigma_2 term
of Eq. (11) an additive constant that cancels out of the argmax. The obstacle
criterion therefore contributed nothing to command selection.

Fix: seed with +inf so the term measures true minimum clearance.

--------------------------------------------------------------------------
FIX 2 -- safety margin inconsistent with the metric      (11.py line 521)
--------------------------------------------------------------------------
DWA rejected a forward-simulated trajectory only at

    uav.radius + obs_r + 0.3

whereas Section 5.2 scores a collision at

    uav.radius + obs_r + 0.5

so DWA regarded a 0.3-0.5 m band as flyable while the evaluation criterion
already counted it as a crash. Fix: use the same 0.5 m margin in both places.

--------------------------------------------------------------------------
FIX 3 -- spline smoothing discarded the collision check  (11.py lines 407-439)
--------------------------------------------------------------------------
`remove_redundant` validates each shortcut with a line-of-sight test
(`is_line_safe`), then `cubic_spline_smooth` replaces those straight segments
with a curve that is never re-checked. Because the greedy shortcut typically
collapses ~48 A* waypoints down to 3-5, the resulting spline bulges well away
from the validated polyline: measured minimum static clearance was +1.94 m along
the polyline but -2.56 m along the spline. The smoother silently undid the very
guarantee Section 4.2.1 claims for it.

Fix: cap the shortcut length so enough waypoints survive to constrain the
spline, then verify the interpolated curve; retry with progressively tighter
caps and fall back to the (already LOS-validated) polyline if none is clear.
This preserves the two-stage design of Section 4.2 and only enforces the
constraint that section already states.
"""

import math

import numpy as np

SAFETY_MARGIN = 0.5  # must match the collision criterion of Section 5.2
SHORTCUT_CAPS = (float("inf"), 30.0, 20.0, 12.0, 8.0)


# --------------------------------------------------------------------------
# FIX 1 + FIX 2
# --------------------------------------------------------------------------
def _evaluate(self, traj, local_goal, global_path, obstacles):
    traj_end = traj[-1]
    dist_goal = math.hypot(traj_end[0] - local_goal[0], traj_end[1] - local_goal[1])

    heading_to_goal = math.atan2(local_goal[1] - self.uav.pos[1],
                                 local_goal[0] - self.uav.pos[0])
    traj_heading = math.atan2(traj_end[1] - traj[0][1], traj_end[0] - traj[0][0])
    heading_diff = abs(heading_to_goal - traj_heading)
    heading_diff = min(heading_diff, 2 * math.pi - heading_diff)
    heading_score = 1.0 - heading_diff / math.pi

    min_obs_dist = float("inf")          # FIX 1: was 0.0
    for p in traj:
        for (ox, oy, obs_r) in obstacles:
            d = math.hypot(p[0] - ox, p[1] - oy) - self.uav.radius - obs_r
            if d < min_obs_dist:
                min_obs_dist = d
    if min_obs_dist == float("inf"):
        min_obs_dist = 99.0

    safe_margin = 3.0
    obs_score = (1.0 - min_obs_dist / safe_margin) if min_obs_dist < safe_margin else 0.0

    w_heading, w_obs = 1.0, 2.0
    w_goal = 0.5 / (dist_goal + 1.0)
    return w_heading * heading_score - w_obs * obs_score + w_goal


def _simulate_with_collision_check(self, v, omega, obstacles):
    if not obstacles:
        return [(self.uav.pos[0], self.uav.pos[1]),
                (self.uav.pos[0] + v * 0.3, self.uav.pos[1] + v * 0.3)]

    traj = [(self.uav.pos[0], self.uav.pos[1])]
    x, y = float(self.uav.pos[0]), float(self.uav.pos[1])
    heading = float(self.uav.heading)
    # FIX 7: ramp the velocity/yaw-rate toward the commanded values subject to
    # the UAV's acceleration limits, so the forward simulation matches the
    # UAV's actual constrained dynamics.
    v_cur = self.uav.v
    w_cur = self.uav.omega
    a_max = getattr(self.uav, "max_accel", float("inf"))
    wa_max = getattr(self.uav, "max_ang_accel", float("inf"))

    for _ in range(int(self.predict_time / self.dt)):
        v_cur += max(-a_max * self.dt, min(a_max * self.dt, v - v_cur))
        w_cur += max(-wa_max * self.dt, min(wa_max * self.dt, omega - w_cur))
        x += v_cur * math.cos(heading) * self.dt
        y += v_cur * math.sin(heading) * self.dt
        heading += w_cur * self.dt
        traj.append((x, y))
        for (ox, oy, obs_r) in obstacles:
            # FIX 2: margin raised from 0.3 to 0.5 to match Section 5.2
            if math.hypot(x - ox, y - oy) < self.uav.radius + obs_r + SAFETY_MARGIN:
                return None
    return traj


# --------------------------------------------------------------------------
# FIX 3
# --------------------------------------------------------------------------
def spline_is_clear(points, env, robot_radius):
    """True if no sampled point of the curve violates the collision criterion."""
    for x, y in points:
        for obs in env.static_obstacles:
            if math.hypot(x - obs.x, y - obs.y) < robot_radius + obs.radius + SAFETY_MARGIN:
                return False
    return True


def remove_redundant_capped(path, env, is_line_safe, risk_threshold=0.6,
                            max_segment=float("inf")):
    """`remove_redundant` of 11.py, with an upper bound on shortcut length.

    Bounding the jump keeps enough control points for the spline to track the
    validated polyline instead of bowing away from it.
    """
    if len(path) < 3:
        return path
    optimized = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        advanced = False
        while j > i + 1:
            seg = math.hypot(path[j][0] - path[i][0], path[j][1] - path[i][1])
            if seg <= max_segment and is_line_safe(env, path[i], path[j], risk_threshold):
                optimized.append(path[j])
                i = j
                advanced = True
                break
            j -= 1
        if not advanced:
            i += 1
            if i < len(path):
                optimized.append(path[i])
    return optimized


def safe_smooth(raw_path, env, robot_radius, smoother_cls, risk_threshold=0.7,
                num_points=150):
    """Two-stage smoothing that verifies its own output.

    Returns (path, n_control_points, cap_used). `cap_used` is None when every
    cap produced an unsafe curve and the polyline fallback was taken.
    """
    reduced = raw_path
    for cap in SHORTCUT_CAPS:
        reduced = remove_redundant_capped(raw_path, env, smoother_cls.is_line_safe,
                                          risk_threshold, cap)
        curve = smoother_cls.cubic_spline_smooth(reduced, num_points=num_points)
        if spline_is_clear(curve, env, robot_radius):
            return curve, len(reduced), cap
    return reduced, len(reduced), None


def apply_all(algorithms_module):
    """Monkey-patch the DWA methods of the imported 11.py module."""
    dwa_cls = algorithms_module.DynamicWindowApproach
    dwa_cls._evaluate = _evaluate
    dwa_cls._simulate_with_collision_check = _simulate_with_collision_check
    # FIX 4: the Kalman filter never ran its time-update (predict) step.
    algorithms_module.Environment.update_kf_trackers = _update_kf_trackers

    # FIX 5: the tracker was seeded with the ground-truth obstacle velocity.
    # An onboard tracker has no such prior; it must estimate velocity from
    # position measurements, so initialise with a zero-velocity prior.
    Obstacle = algorithms_module.Obstacle
    ObstacleTracker = algorithms_module.ObstacleTracker

    def _add_dynamic_obstacle(self, x, y, radius, vx, vy, risk_coeff=1.0):
        obs = Obstacle(x, y, radius, risk_coeff, is_dynamic=True, vx=vx, vy=vy)
        self.dynamic_obstacles.append(obs)
        tracker = ObstacleTracker(len(self.obstacle_trackers), dt=0.2)
        tracker.kf.initialize(x, y, 0.0, 0.0)
        self.obstacle_trackers.append(tracker)

    algorithms_module.Environment.add_dynamic_obstacle = _add_dynamic_obstacle

    # FIX 6: the Kalman filter's noise variances were not configurable
    # (process_noise was hardcoded to 0.5 inside predict()). Store them as
    # instance attributes so the sensitivity study (Review 2 #34) can vary them.
    algorithms_module.KalmanFilter.__init__ = _kalman_init
    algorithms_module.KalmanFilter.predict = _kalman_predict
    return algorithms_module


# --------------------------------------------------------------------------
# FIX 6 -- Kalman filter noise parameters were not configurable
# --------------------------------------------------------------------------
# `predict()` hardcoded the process-noise variance to 0.5 when rebuilding Q, so
# the Q matrix could not be changed after construction. Store both noise
# variances as instance attributes so they can be varied for the sensitivity
# study. Behaviour at the nominal values (0.5, 2.0) is unchanged.


def _kalman_init(self, dt, process_noise=0.5, measurement_noise=2.0):
    self.dt = dt
    self.process_noise = process_noise
    self.measurement_noise = measurement_noise
    self.x = np.zeros(4)
    self.P = np.eye(4) * 5.0
    self.F = np.array([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    self.H = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]])
    self.Q = np.eye(4) * process_noise
    self.Q[0, 0] = process_noise * dt ** 4 / 4
    self.Q[1, 1] = process_noise * dt ** 4 / 4
    self.Q[2, 2] = process_noise * dt ** 2
    self.Q[3, 3] = process_noise * dt ** 2
    self.R = np.eye(2) * measurement_noise
    self.initialized = False


def _kalman_predict(self, dt=None):
    if dt is not None:
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        pn = self.process_noise
        self.Q = np.eye(4) * pn
        self.Q[0, 0] = pn * dt ** 4 / 4
        self.Q[1, 1] = pn * dt ** 4 / 4
        self.Q[2, 2] = pn * dt ** 2
        self.Q[3, 3] = pn * dt ** 2
    self.x = self.F @ self.x
    self.P = self.F @ self.P @ self.F.T + self.Q


# --------------------------------------------------------------------------
# FIX 4 -- Kalman filter never performed the time-update (predict) step
# --------------------------------------------------------------------------
# `Environment.update_kf_trackers` only applied the measurement update
# (`kf.update`), never the time update (`kf.predict`). A Kalman filter that
# never propagates its motion model degenerates into a static smoother: the
# velocity estimate stays frozen at its initial value and the position estimate
# lags a moving obstacle by ~12 m over a 40 s run (see diagnose_kf.py). The
# manuscript's Section 3.3 states the filter "executes prediction and update
# steps in each cycle", so the code omitted the very step the paper describes.
#
# Fix: propagate the state forward (predict) before each measurement update,
# yielding a proper constant-velocity tracker.
def _update_kf_trackers(self, current_time):
    for obs, tracker in zip(self.dynamic_obstacles, self.obstacle_trackers):
        tracker.predict(tracker.kf.dt)  # time update: x = F x, P = F P F' + Q
        tracker.update_observation(obs, current_time)
