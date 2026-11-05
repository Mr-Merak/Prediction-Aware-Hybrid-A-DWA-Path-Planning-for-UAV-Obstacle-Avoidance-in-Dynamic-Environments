"""
failure_analysis.py -- Reviewer 2 suggestion #5: failure-mode analysis.

Re-runs the three DWA-based configurations on the nominal scene and records, for
every collision, the relative geometry between the UAV and the colliding obstacle,
classifying each collision as head-on / crossing / overtaking / static according to
the angle between the UAV velocity vector and the obstacle velocity vector:

    |angle| < 45 deg          -> overtaking (same direction, one catches the other)
    45 <= |angle| <= 135 deg  -> crossing   (roughly perpendicular paths)
    |angle| > 135 deg         -> head-on    (approaching from opposite directions)

Outputs a per-method collision-by-category table and a figure of representative
failure trajectories (one panel per category).

Run:
    py failure_analysis.py --runs 100 --jobs 7
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import benchmark as B

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
FIG_DIR = os.path.dirname(HERE)  # figures live at repo root next to the tex

CONFIGS = ["astar_dwa", "proposed", "proposed_pred_dwa"]


def find_collision(env, uav):
    """Return the colliding obstacle (dynamic preferred) or None."""
    best = None
    for obs in env.dynamic_obstacles:
        d = math.hypot(uav.pos[0] - obs.x, uav.pos[1] - obs.y)
        if d < uav.radius + obs.radius + 0.5:
            return ("dynamic", obs)
    for obs in env.static_obstacles:
        d = math.hypot(uav.pos[0] - obs.x, uav.pos[1] - obs.y)
        if d < uav.radius + obs.radius + 0.5:
            return ("static", obs)
    return None


def classify_collision(uav, obs):
    """Classify a dynamic-obstacle collision by velocity-vector angle (degrees)."""
    if obs is None:
        return "static"
    ux = uav.v * math.cos(uav.heading)
    uy = uav.v * math.sin(uav.heading)
    ox, oy = obs.vx, obs.vy
    nu = math.hypot(ux, uy)
    no = math.hypot(ox, oy)
    if nu < 1e-6 or no < 1e-6:
        return "static"  # degenerate (essentially stationary)
    cosang = max(-1.0, min(1.0, (ux * ox + uy * oy) / (nu * no)))
    ang = math.degrees(math.acos(cosang))
    if ang < 45.0:
        return "overtaking"
    if ang <= 135.0:
        return "crossing"
    return "head-on"


def recorded_run(cfg_name, seed):
    """Faithful copy of benchmark.run_episode with trajectory + collision recording."""
    cfg = B.CONFIGS[cfg_name]
    use_kf = cfg["kf"]
    use_dwa = cfg["dwa"]
    use_smooth = cfg["smooth"]
    risk_switch = cfg.get("risk_switch", True)
    euclid_only = cfg.get("euclid_only", False) or not cfg["adaptive"]
    astar_risk = cfg["astar_risk"] and not cfg.get("euclid_only", False)
    planner_kind = cfg.get("planner", "astar")

    env, uav = B.build_scenario(seed, False)
    smoother = B.A.PathSmoother()
    if cfg.get("pred_dwa", False):
        dwa = B.PredictionAwareDWA(uav, env, B.GOAL)
    else:
        dwa = B.A.DynamicWindowApproach(uav, env, B.GOAL)

    if planner_kind == "dstar":
        raise NotImplementedError("failure analysis covers A*-based configs only")

    if not cfg["adaptive"] or cfg.get("euclid_only", False):
        planner_cls = B.RiskBlindAStar if not astar_risk else B.EuclideanAStar
    else:
        planner_cls = B.A.AdaptiveAStar

    def do_replan():
        env.update_kf_trackers(sim_time)
        env.compute_risk_grid(uav, sim_time, use_kf=use_kf, n_steps=B.N_PRED_STEPS)
        planner = planner_cls(env, uav, B.GOAL,
                              risk_threshold=B.ASTAR_RISK_THRESHOLD, k=B.BETA)
        raw = planner.run()
        if use_smooth:
            out, _nc, _cap = B.fixes.safe_smooth(
                raw, env, uav.radius, B.A.PathSmoother,
                risk_threshold=B.SMOOTH_RISK_THRESHOLD, num_points=150)
        else:
            out = raw
        return out

    sim_time = 0.0
    env.compute_risk_grid(uav, 0.0, use_kf=use_kf, n_steps=B.N_PRED_STEPS)
    path = do_replan()

    last_replan = 0.0
    dwa_active = False
    dwa_start = 0.0
    consecutive_high = 0

    traj = [(float(uav.pos[0]), float(uav.pos[1]))]
    obstacle_traj = {i: [(o.x, o.y)] for i, o in enumerate(env.dynamic_obstacles)}
    collision_info = None

    outcome = "timeout"
    n_steps = int(B.MAX_SIM_TIME / B.DT)

    for _ in range(n_steps):
        sim_time += B.DT
        env.update_dynamic(B.DT)
        env.update_kf_trackers(sim_time)
        env.compute_risk_grid(uav, sim_time, use_kf=use_kf, n_steps=B.N_PRED_STEPS)
        env.compute_eval_risk_grid(uav, n_steps=B.N_PRED_STEPS)

        for i, o in enumerate(env.dynamic_obstacles):
            obstacle_traj[i].append((o.x, o.y))

        hit = find_collision(env, uav)
        if hit is not None:
            kind, obs = hit
            collision_info = dict(
                category=(classify_collision(uav, obs) if kind == "dynamic" else "static"),
                kind=kind,
                uav_pos=(uav.pos[0], uav.pos[1]),
                uav_vel=(uav.v * math.cos(uav.heading), uav.v * math.sin(uav.heading)),
                obs_vel=(obs.vx, obs.vy) if kind == "dynamic" else (0.0, 0.0),
                time=sim_time)
            outcome = "collision"
            break
        if uav.distance_to(B.GOAL) < B.GOAL_TOLERANCE:
            outcome = "success"
            break

        replan_trigger = (sim_time - last_replan) >= B.REPLAN_INTERVAL
        if use_dwa and dwa_active and (sim_time - dwa_start > 3.0):
            if B.planner_risk_box(env, uav) < 0.15:
                dwa_active = False
                replan_trigger = True
        if replan_trigger and planner_kind != "dwa_only":
            path = do_replan()
            last_replan = sim_time

        local_risk = B.planner_risk_box(env, uav)
        if use_dwa and risk_switch:
            if local_risk > B.RISK_TRIGGER:
                consecutive_high += 1
            else:
                consecutive_high = 0
            if consecutive_high >= 2 and not dwa_active:
                dwa_active = True
                dwa_start = sim_time
                consecutive_high = 0

        if use_dwa:
            if dwa_active:
                target = B.find_lookahead(path, uav.pos, 3.0, 10)
                v, omega = dwa.calculate_velocity(path, target)
            else:
                target = B.find_lookahead(path, uav.pos, 8.0, 25)
                v, omega = dwa.calculate_velocity(path, target)
        else:
            target = B.find_lookahead(path, uav.pos, 8.0, 25)
            v, omega = dwa._go_towards(target, uav.max_speed * 0.8)

        uav.update(v, omega, B.DT)
        traj.append((float(uav.pos[0]), float(uav.pos[1])))

        if uav.stuck_timer > 20.0:
            outcome = "stuck"
            break

    return dict(config=cfg_name, seed=seed, outcome=outcome,
                collision=collision_info, traj=traj, obstacle_traj=obstacle_traj)


def _worker(job):
    cfg, seed = job
    try:
        return recorded_run(cfg, seed)
    except Exception as exc:
        return dict(config=cfg, seed=seed, outcome="error:%s" % exc,
                    collision=None, traj=[], obstacle_traj={})


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "experiment_data_failure.xlsx"))
    args = ap.parse_args()

    jobs_list = [(c, s) for c in CONFIGS for s in range(args.runs)]
    print("Failure analysis: %d runs x %d configs on nominal scene | %d workers"
          % (args.runs, len(CONFIGS), args.jobs))
    t0 = time.time()
    rows = []
    with mp.Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs_list), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(jobs_list):
                print("  %d/%d (%.0f s)" % (i, len(jobs_list), time.time() - t0), flush=True)

    # ---- aggregate collision categories ----
    cats = ["head-on", "crossing", "overtaking", "static"]
    summary = {}
    for cfg in CONFIGS:
        rs = [r for r in rows if r["config"] == cfg]
        n = len(rs)
        succ = sum(1 for r in rs if r["outcome"] == "success")
        counts = {c: 0 for c in cats}
        examples = {c: None for c in cats}  # seed of a representative example
        for r in rs:
            if r["outcome"] == "collision" and r["collision"]:
                c = r["collision"]["category"]
                counts[c] = counts.get(c, 0) + 1
                if examples.get(c) is None:
                    examples[c] = r["seed"]
        summary[cfg] = dict(n=n, success=succ, counts=counts, examples=examples)

    print("\n===== Collision category counts (nominal scene) =====")
    print("%-28s %8s %8s %8s %8s %8s" % ("Configuration", "Succ", "Head-on",
                                          "Cross", "Over-tk", "Static"))
    for cfg in CONFIGS:
        s = summary[cfg]
        print("%-28s %7d %8d %8d %8d %8d"
              % (B.CONFIGS[cfg]["label"], s["success"], s["counts"]["head-on"],
                 s["counts"]["crossing"], s["counts"]["overtaking"],
                 s["counts"]["static"]))

    # ---- write xlsx ----
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment
    wb = Workbook()
    bold = Font(bold=True)
    ws = wb.active
    ws.title = "Collision categories"
    ws.append(["Configuration", "Runs", "Successes", "Head-on", "Crossing",
               "Overtaking", "Static", "Total collisions"])
    for c in ws[1]:
        c.font = bold
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for cfg in CONFIGS:
        s = summary[cfg]
        total = sum(s["counts"].values())
        ws.append([B.CONFIGS[cfg]["label"], s["n"], s["success"],
                   s["counts"]["head-on"], s["counts"]["crossing"],
                   s["counts"]["overtaking"], s["counts"]["static"], total])
    for col in "ABCDEFGH":
        ws.column_dimensions[col].width = 16

    ws2 = wb.create_sheet("Raw runs")
    ws2.append(["Configuration", "Seed", "Outcome", "Collision category",
                "Collision time (s)", "UAV vx", "UAV vy", "Obs vx", "Obs vy"])
    for c in ws2[1]:
        c.font = bold
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for r in sorted(rows, key=lambda r: (CONFIGS.index(r["config"]), r["seed"])):
        ci = r["collision"] or {}
        uv = ci.get("uav_vel", (float("nan"), float("nan")))
        ov = ci.get("obs_vel", (float("nan"), float("nan")))
        ws2.append([B.CONFIGS[r["config"]]["label"], r["seed"], r["outcome"],
                    ci.get("category", ""), round(ci.get("time", float("nan")), 1) if ci else "",
                    round(uv[0], 3), round(uv[1], 3), round(ov[0], 3), round(ov[1], 3)])
    for col in "ABCDEFGHI":
        ws2.column_dimensions[col].width = 16
    wb.save(args.out)
    print("\nWrote %s" % args.out)

    # ---- representative failure trajectories ----
    _plot_failures(rows, summary, CONFIGS)


def _plot_failures(rows, summary, configs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = ["head-on", "crossing", "overtaking"]
    by_seed = {}
    for r in rows:
        by_seed[(r["config"], r["seed"])] = r

    fig, axes = plt.subplots(3, len(configs), figsize=(4.2 * len(configs), 11),
                             constrained_layout=True)
    for ci, cfg in enumerate(configs):
        for ri, cat in enumerate(cats):
            ax = axes[ri, ci]
            seed = summary[cfg]["examples"].get(cat)
            r = by_seed.get((cfg, seed)) if seed is not None else None
            if r is None:
                ax.set_title("%s\n%s: no example" % (B.CONFIGS[cfg]["label"], cat))
                ax.axis("off")
                continue
            # static obstacles
            env_dummy = None
            traj = r["traj"]
            # plot UAV trajectory
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, "-", color="C0", lw=1.6, label="UAV")
            ax.plot(xs[0], ys[0], "o", color="C0", ms=6)
            ax.plot(xs[-1], ys[-1], "x", color="C3", ms=9, mew=2)
            # plot all dynamic obstacle trajectories (faint) and the colliding one (bold)
            colliding = None
            if r["collision"] and r["collision"]["kind"] == "dynamic":
                # find obstacle index nearest the collision point
                cx, cy = r["collision"]["uav_pos"]
                dmin, colliding = 1e18, None
                for i, ot in r["obstacle_traj"].items():
                    ox, oy = ot[-1]
                    d = math.hypot(ox - cx, oy - cy)
                    if d < dmin:
                        dmin, colliding = d, i
            for i, ot in r["obstacle_traj"].items():
                oxs = [p[0] for p in ot]
                oys = [p[1] for p in ot]
                if i == colliding:
                    ax.plot(oxs, oys, "-", color="C2", lw=2.0, label="colliding obs")
                else:
                    ax.plot(oxs, oys, "-", color="0.8", lw=0.7)
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 200)
            ax.set_aspect("equal")
            ax.set_title("%s\n%s (seed %d)"
                         % (B.CONFIGS[cfg]["label"], cat, seed), fontsize=8)
            ax.set_xlabel("x (m)", fontsize=7)
            ax.set_ylabel("y (m)", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle("Representative failure trajectories by collision geometry", fontsize=11)
    out_path = os.path.join(FIG_DIR, "figure_failure_modes.png")
    fig.savefig(out_path, dpi=130)
    print("Wrote %s" % out_path)


if __name__ == "__main__":
    main()
