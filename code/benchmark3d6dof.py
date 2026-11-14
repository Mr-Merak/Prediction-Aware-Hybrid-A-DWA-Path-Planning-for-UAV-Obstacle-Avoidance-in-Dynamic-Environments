"""
benchmark3d6dof.py -- full 6-DOF quadrotor validation of the 3D extension.

This is the strongest validation available in the current revision cycle in
place of a Gazebo/ROS2 or flight campaign (Expert 1, round 2).  The 3D
planning stack of benchmark3d.py is kept, but the kinematic point-mass UAV is
replaced by a rigid-body quadrotor with:

  * 6-DOF rigid-body dynamics (position, velocity, SO(3) attitude, body rates),
  * a cascaded position/velocity -> attitude controller,
  * thrust and torque saturation, tilt/acceleration limits,
  * optional constant wind disturbance.

Two full-3D scenarios are evaluated:
  * "window": the altitude-window scene of Section 5.11, with dynamic obstacles
    now moving in all three dimensions (vz != 0);
  * "crossing3d": dense crossing traffic at different altitudes, with both
    horizontal and vertical movers and no static walls.

Baselines: a true Euclidean 3D A* (not the risk-adaptive planner), the
risk-adaptive 3D A*, and the two DWA variants.  All configurations use the
same 6-DOF plant and the same paired seeds.

Run:
    py benchmark3d6dof.py --runs 50 --jobs 7
"""
import argparse
import importlib.util
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_benchmark3d():
    path = os.path.join(HERE, "benchmark3d.py")
    spec = importlib.util.spec_from_file_location("benchmark3d", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark3d"] = mod
    spec.loader.exec_module(mod)
    return mod


B3 = _load_benchmark3d()

DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

CONFIGS = ["astar3d", "ia3d", "ia3d_dwa", "ia3d_pred_dwa"]
LABELS = {
    "astar3d": "3D A* (Euclidean, pure pursuit)",
    "ia3d": "3D IA* (risk-adaptive, pure pursuit)",
    "ia3d_dwa": "3D IA* + DWA",
    "ia3d_pred_dwa": "3D IA* + prediction-aware DWA",
}
SCENARIOS = ["window", "crossing3d"]
SCENARIO_LABELS = {
    "window": "Altitude window (3D movers)",
    "crossing3d": "3D crossing traffic",
}

# --------------------------------------------------------------------------
# Quadrotor parameters and gains
# --------------------------------------------------------------------------
MASS = 1.5
G = 9.81
J = np.diag([0.02, 0.02, 0.04])
J_INV = np.linalg.inv(J)
T_MAX = 1.8 * MASS * G          # ~26.5 N
TAU_MAX = np.array([0.35, 0.35, 0.15])
A_MAX_XY = 4.0                  # m/s^2
A_MAX_Z = 3.0
KP_V = 2.2                      # velocity-loop gain (1/s)
KP_R = 8.0                      # attitude gain
KD_R = 2.4                      # body-rate gain
SUB_DT = 0.01                   # controller/integration step
WIND = np.zeros(3)              # optional constant wind acceleration (m/s^2)

# Global-only baselines track the A* path with pure pursuit.  A 2.0 m/s
# cruise speed cuts the corner at the altitude window and crashes even on a
# collision-free path; 1.6 m/s with the standard 6 m look-ahead tracks the
# same feasible path reliably.  The DWA configurations are unaffected.
PURE_PURSUIT_SPEED = 1.6        # m/s
PURE_PURSUIT_LOOKAHEAD = 6.0    # m


def _skew(w):
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def _vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def _Rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _R_from_z_yaw(z_des, yaw):
    z_b = z_des / max(np.linalg.norm(z_des), 1e-9)
    x_c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y_b = np.cross(z_b, x_c)
    if np.linalg.norm(y_b) < 1e-6:
        x_c = np.array([math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2), 0.0])
        y_b = np.cross(z_b, x_c)
    y_b /= np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)
    return np.column_stack([x_b, y_b, z_b])


def _clip_norm(v, lim):
    n = np.linalg.norm(v)
    if n > lim and n > 0:
        return v * (lim / n)
    return v


class Quad6DOF:
    """Rigid-body quadrotor with the same interface as benchmark3d.UAV3D."""

    def __init__(self, p, yaw=None):
        self.p = np.array(p, dtype=float)
        if yaw is None:
            yaw = math.atan2(B3.GOAL[1] - p[1], B3.GOAL[0] - p[0])
        self.R = _Rz(yaw)
        self.v_world = np.zeros(3)
        self.omega_body = np.zeros(3)
        self.yaw_ref = yaw
        # interface attributes read by the planners / metrics
        self.v = 0.0
        self.omega = 0.0
        self.vz = 0.0

    @property
    def x(self):
        return float(self.p[0])

    @property
    def y(self):
        return float(self.p[1])

    @property
    def z(self):
        return float(self.p[2])

    @property
    def heading(self):
        return math.atan2(self.R[1, 0], self.R[0, 0])

    def dist(self, p):
        return float(np.linalg.norm(self.p - np.asarray(p, dtype=float)))

    def update(self, v_cmd, omega_cmd, vz_cmd, dt):
        yaw = self.heading
        # yaw reference: apply the planner's yaw-rate command over this step
        self.yaw_ref = yaw + float(omega_cmd) * dt
        v_des = np.array([float(v_cmd) * math.cos(yaw),
                          float(v_cmd) * math.sin(yaw),
                          float(vz_cmd)])

        n_sub = max(1, int(round(dt / SUB_DT)))
        h = dt / n_sub
        for _ in range(n_sub):
            # ---- velocity loop: desired world acceleration -> thrust vector ----
            e_v = v_des - self.v_world
            a_des = KP_V * e_v
            a_des[:2] = _clip_norm(a_des[:2], A_MAX_XY)
            a_des[2] = max(-A_MAX_Z, min(A_MAX_Z, a_des[2]))
            f_des = MASS * (a_des + np.array([0.0, 0.0, G]))
            thrust = min(max(float(np.linalg.norm(f_des)), 0.0), T_MAX)
            if np.linalg.norm(f_des) < 1e-9:
                z_des = np.array([0.0, 0.0, 1.0])
            else:
                z_des = f_des / np.linalg.norm(f_des)
            R_des = _R_from_z_yaw(z_des, self.yaw_ref)

            # ---- attitude loop ----
            e_R = 0.5 * _vee(R_des.T @ self.R - self.R.T @ R_des)
            tau = -KP_R * e_R - KD_R * self.omega_body \
                  + np.cross(self.omega_body, J @ self.omega_body)
            tau = np.clip(tau, -TAU_MAX, TAU_MAX)

            # ---- rigid-body integration ----
            a_world = self.R @ np.array([0.0, 0.0, thrust / MASS]) \
                      - np.array([0.0, 0.0, G]) + WIND
            self.v_world += a_world * h
            self.p += self.v_world * h
            Rdot = self.R @ _skew(self.omega_body)
            self.R = self.R + Rdot * h
            U, _, Vt = np.linalg.svd(self.R)
            self.R = U @ Vt
            self.omega_body += J_INV @ (tau - np.cross(self.omega_body,
                                                        J @ self.omega_body)) * h

        # keep the vehicle inside the volume / above the ground
        self.p[2] = max(0.2, min(B3.BOUNDS[5] - 0.2, self.p[2]))
        self.v = float(np.linalg.norm(self.v_world[:2]))
        self.omega = float(self.omega_body[2])
        self.vz = float(self.v_world[2])


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------
def _free_of_static(env, x, y, z, r):
    for (sx, sy, sz, sr) in env.static:
        if math.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2) < r + sr + 1.5:
            return False
    return True


def build_env_6dof(seed, scenario):
    env = B3.Env3D()
    rng = np.random.default_rng(seed)

    if scenario == "window":
        # Same altitude-window walls as Section 5.11 (benchmark3d.build_env).
        for yy in np.arange(-4.0, 104.0, 2.5):
            for zz in np.arange(2.0, 21.0, 3.0):
                env.add_static(50.0, yy, zz, 2.0)
            for zz in np.arange(32.0, 41.0, 3.0):
                env.add_static(50.0, yy, zz, 2.0)
        env.build_static_grid()
        # Three vertical movers in the approach corridors, crossing the altitude
        # window that the UAV must fly through (a genuinely 3D threat).
        for i in range(3):
            x = 43.0 if i % 2 == 0 else 57.0
            y = float(rng.uniform(38, 62))
            z = float(rng.uniform(10, 32))
            r = 2.0
            vz = (1.0 if i < 2 else -1.0) * float(rng.uniform(1.0, 1.5))
            env.add_dynamic(x, y, z, r, float(rng.uniform(-0.2, 0.2)),
                            float(rng.uniform(-0.2, 0.2)), vz)
        # Four horizontal movers at window altitude, approaching the gap.
        for i in range(4):
            x = float(rng.uniform(15, 40) if i < 2 else rng.uniform(60, 85))
            y = float(rng.uniform(25, 75))
            z = float(rng.uniform(21, 29))
            ang = float(rng.uniform(0, 2 * math.pi)); sp = float(rng.uniform(1.2, 1.7))
            env.add_dynamic(x, y, z, 2.0, sp * math.cos(ang), sp * math.sin(ang),
                            float(rng.uniform(-0.2, 0.2)))

    elif scenario == "crossing3d":
        # No static walls; dense 3D traffic at several altitudes.  Half of the
        # movers are primarily horizontal, half primarily vertical.
        env.build_static_grid()
        for i in range(8):
            for _attempt in range(200):
                x = float(rng.uniform(10, 90)); y = float(rng.uniform(20, 80))
                z = float(rng.uniform(5, 35)); r = 2.0
                if _free_of_static(env, x, y, z, r):
                    break
            if i % 2 == 0:
                sp = float(rng.uniform(1.2, 1.8)); ang = float(rng.uniform(0, 2 * math.pi))
                vx, vy, vz = sp * math.cos(ang), sp * math.sin(ang), float(rng.uniform(-0.3, 0.3))
            else:
                vx = float(rng.uniform(-0.6, 0.6)); vy = float(rng.uniform(-0.6, 0.6))
                vz = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.8, 1.3))
            env.add_dynamic(x, y, z, r, vx, vy, vz)
    else:
        raise ValueError("unknown scenario: %r" % (scenario,))

    return env


# --------------------------------------------------------------------------
# Episode
# --------------------------------------------------------------------------
def run_episode(cfg, seed, scenario):
    env = build_env_6dof(seed, scenario)
    uav = Quad6DOF(B3.START)
    risk_adaptive = cfg != "astar3d"
    use_dwa = cfg in ("ia3d_dwa", "ia3d_pred_dwa")
    dwa = B3.DWA3D(env, uav, B3.GOAL, pred=(cfg == "ia3d_pred_dwa"))

    t_wall0 = time.perf_counter()
    sim_time = 0.0
    env.compute_risk(use_kf=True)
    planner = B3.AStar3D(env, B3.START, B3.GOAL, risk_adaptive=risk_adaptive)
    path = planner.run()

    last_replan = 0.0
    outcome = "timeout"
    traj = [(uav.x, uav.y, uav.z)]
    path_len = 0.0
    prev = np.array([uav.x, uav.y, uav.z])
    track_errors = []
    replan_times = []
    n_replans = 0
    n_steps = int(B3.MAX_T / B3.DT)

    for _ in range(n_steps):
        sim_time += B3.DT
        env.step_dynamic()
        env.update_kf()

        if env.is_collision(uav.x, uav.y, uav.z, B3.UAV_R):
            outcome = "collision"
            break
        if uav.z < 0.25:
            outcome = "ground"
            break
        if uav.dist(B3.GOAL) < B3.GOAL_TOL:
            outcome = "success"
            break

        if (sim_time - last_replan) >= B3.REPLAN:
            t0 = time.perf_counter()
            env.compute_risk(use_kf=True)
            planner = B3.AStar3D(env, (uav.x, uav.y, uav.z), B3.GOAL,
                                 risk_adaptive=risk_adaptive)
            new_path = planner.run()
            if new_path:
                path = new_path
            last_replan = sim_time
            replan_times.append(time.perf_counter() - t0)
            n_replans += 1

        if path is None:
            path = planner.run() or [B3.GOAL]

        if use_dwa:
            v, omega, vz = dwa.compute(path)
        else:
            # pure-pursuit tracking of the global path (same as benchmark3d)
            target = B3.find_lookahead3d(path, uav, PURE_PURSUIT_LOOKAHEAD, 30)
            dx, dy, dz = target[0] - uav.x, target[1] - uav.y, target[2] - uav.z
            dhead = math.atan2(dy, dx) - uav.heading
            dhead = math.atan2(math.sin(dhead), math.cos(dhead))
            v = PURE_PURSUIT_SPEED
            omega = max(-B3.OMEGA_MAX, min(B3.OMEGA_MAX, 2.0 * dhead))
            vz = max(-B3.VZ_MAX, min(B3.VZ_MAX, dz * 1.0))

        uav.update(v, omega, vz, B3.DT)

        cur = np.array([uav.x, uav.y, uav.z])
        path_len += float(np.linalg.norm(cur - prev))
        prev = cur
        traj.append((uav.x, uav.y, uav.z))
        if path:
            track_errors.append(min(
                math.sqrt((uav.x - p[0]) ** 2 + (uav.y - p[1]) ** 2 + (uav.z - p[2]) ** 2)
                for p in path))

    success = outcome == "success"
    wall = time.perf_counter() - t_wall0
    return dict(config=cfg, seed=seed, scenario=scenario, success=int(success),
                outcome=outcome,
                path_length=path_len if success else float("nan"),
                tracking_error=float(np.mean(track_errors)) if (success and track_errors) else float("nan"),
                sim_time=sim_time,
                full_step_time=wall / max(1, int(round(sim_time / B3.DT))),
                replan_time=float(np.mean(replan_times)) if replan_times else float("nan"),
                n_replans=n_replans,
                final_dist=uav.dist(B3.GOAL),
                traj=traj)


def _worker(job):
    cfg, seed, scenario = job
    try:
        return run_episode(cfg, seed, scenario)
    except Exception as exc:
        return dict(config=cfg, seed=seed, scenario=scenario, success=0,
                    outcome="error:%s" % exc, path_length=float("nan"),
                    tracking_error=float("nan"), sim_time=0.0,
                    full_step_time=float("nan"), replan_time=float("nan"),
                    n_replans=0, final_dist=float("nan"), traj=[])


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _write_xlsx(rows, path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Scenario", "Configuration", "Runs", "Successes", "Success Rate (%)",
               "95% CI Low", "95% CI High", "Path Length (m)", "Tracking Error (m)",
               "Full Step Time (s)", "Replanning Time (s)", "Collisions", "Timeouts", "Ground"])
    for c in ws[1]:
        c.font = Font(bold=True)
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for scenario in SCENARIOS:
        for cfg in CONFIGS:
            rs = [r for r in rows if r["scenario"] == scenario and r["config"] == cfg]
            n = len(rs)
            succ = sum(1 for r in rs if r["success"])
            lo, hi = wilson_ci(succ, n)
            ps = [r for r in rs if r["success"] and not math.isnan(r["path_length"])]
            ts = [r for r in rs if r["success"] and not math.isnan(r["tracking_error"])]
            rt = [r["replan_time"] for r in rs if not math.isnan(r["replan_time"])]
            ws.append([SCENARIO_LABELS[scenario], LABELS[cfg], n, succ,
                       round(100.0 * succ / n, 1) if n else None,
                       round(100.0 * lo, 1), round(100.0 * hi, 1),
                       round(float(np.mean([r["path_length"] for r in ps])), 1) if ps else None,
                       round(float(np.mean([r["tracking_error"] for r in ts])), 2) if ts else None,
                       round(float(np.mean([r["full_step_time"] for r in rs])), 4),
                       round(float(np.mean(rt)), 4) if rt else None,
                       sum(1 for r in rs if r["outcome"] == "collision"),
                       sum(1 for r in rs if r["outcome"] == "timeout"),
                       sum(1 for r in rs if r["outcome"] == "ground")])
    for col in "ABCDEFGHIJKLMN":
        ws.column_dimensions[col].width = 22

    ws2 = wb.create_sheet("Raw runs")
    ws2.append(["Scenario", "Configuration", "Seed", "Success", "Outcome",
                "Path Length (m)", "Tracking Error (m)", "Full Step Time (s)",
                "Replanning Time (s)", "Replans", "Final Distance (m)"])
    for r in sorted(rows, key=lambda x: (x["scenario"], x["config"], x["seed"])):
        ws2.append([SCENARIO_LABELS[r["scenario"]], LABELS[r["config"]], r["seed"],
                    r["success"], r["outcome"],
                    None if math.isnan(r["path_length"]) else round(r["path_length"], 2),
                    None if math.isnan(r["tracking_error"]) else round(r["tracking_error"], 2),
                    round(r["full_step_time"], 4),
                    None if math.isnan(r["replan_time"]) else round(r["replan_time"], 4),
                    r["n_replans"], round(r["final_dist"], 2)])
    wb.save(path)


def _save_trajectories(rows, path):
    """Save one representative successful trajectory per scenario/config."""
    import json
    out = {}
    for scenario in SCENARIOS:
        for cfg in CONFIGS:
            rs = [r for r in rows if r["scenario"] == scenario and r["config"] == cfg
                  and r["success"] and r["traj"]]
            if rs:
                out["%s/%s" % (scenario, cfg)] = rs[0]["traj"]
    with open(path, "w") as f:
        json.dump(out, f)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "experiment_data_3d6dof.xlsx"))
    ap.add_argument("--traj", default=os.path.join(DATA_DIR, "trajectories_3d6dof.json"))
    ap.add_argument("--scenarios", default=",".join(SCENARIOS),
                    help="comma-separated subset of: window,crossing3d")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    jobs = [(c, s, sc) for sc in scenarios for c in CONFIGS for s in range(args.runs)]
    print("6-DOF 3D validation: %d scenarios x %d configs x %d seeds = %d episodes | %d workers"
          % (len(scenarios), len(CONFIGS), args.runs, len(jobs), args.jobs))

    t0 = time.time()
    rows = []
    with mp.Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(jobs):
                print("  %d/%d (%.0f s)" % (i, len(jobs), time.time() - t0), flush=True)

    print("\n===== 6-DOF results =====")
    print("%-26s %-36s %7s %-16s %9s %9s" % (
        "Scenario", "Configuration", "Succ%", "95% CI", "Path(m)", "TrkErr"))
    for scenario in scenarios:
        for cfg in CONFIGS:
            rs = [r for r in rows if r["scenario"] == scenario and r["config"] == cfg]
            n = len(rs); succ = sum(1 for r in rs if r["success"])
            lo, hi = wilson_ci(succ, n)
            ps = [r["path_length"] for r in rs if not math.isnan(r["path_length"])]
            ts = [r["tracking_error"] for r in rs if not math.isnan(r["tracking_error"])]
            print("%-26s %-36s %6.1f  [%4.1f,%4.1f] %9.1f %9.2f" % (
                SCENARIO_LABELS[scenario], LABELS[cfg],
                100.0 * succ / n if n else 0.0, 100.0 * lo, 100.0 * hi,
                float(np.mean(ps)) if ps else float("nan"),
                float(np.mean(ts)) if ts else float("nan")))

    _write_xlsx(rows, args.out)
    _save_trajectories(rows, args.traj)
    print("\nWrote %s" % args.out)
    print("Wrote %s" % args.traj)


if __name__ == "__main__":
    main()
