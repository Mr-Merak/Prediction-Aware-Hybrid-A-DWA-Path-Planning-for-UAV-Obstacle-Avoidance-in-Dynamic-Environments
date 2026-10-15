"""
make_figures.py -- Regenerate the result figures with the corrected code.

All figures are produced from the same fixed code (benchmark.py -> 11.py +
fixes.py + tuning.py, including the Kalman predict-step fix, the constrained
UAV kinematics, and prediction-aware DWA) and from experiment_data_v4.xlsx
(100 paired trials). Output filenames match v8_paper.tex:

  figure5.png  risk-field evolution during a proposed run
  figure6.png  representative simulation result (successful run)
  figure7.png  trajectory comparison: A*+DWA vs proposed vs proposed+pred-DWA
  figure8.png  box plots of the 100-run distributions
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import math
import numpy as np

import benchmark as B

PRED_STEPS = 25  # matches the reported experiments (5.0 s horizon)

# Restrained categorical palette (validated all-pairs).
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
STATIC_FC = "#d9d8d4"
STATIC_EC = "#898781"
DYN_FC = "#bcd4f6"
DYN_EC = "#2a78d6"
START_COLOR = "#008300"
GOAL_COLOR = "#0b0b0b"

CONFIG_LABELS = {
    "astar_dwa": "A* + DWA",
    "ia_only": "IA*-only",
    "proposed": "Proposed framework",
    "proposed_pred_dwa": "Proposed + prediction-aware DWA",
    "abl_kf": "w/o KF prediction",
    "abl_adah": "w/o adaptive heuristic",
}


def _make_dwa(cfg, uav, env):
    """Instantiate the correct DWA (prediction-aware or standard) per config."""
    if cfg.get("pred_dwa", False):
        return B.PredictionAwareDWA(uav, env, B.GOAL)
    return B.A.DynamicWindowApproach(uav, env, B.GOAL)


def run_record(cfg_name, seed, snapshot_times=()):
    """Run one episode and record the flown trajectory and risk-field snapshots.

    Mirrors benchmark.run_episode so the visuals are faithful to the reported
    data; only trajectory points and risk grids are captured in addition.
    """
    cfg = B.CONFIGS[cfg_name]
    use_kf = cfg["kf"]
    use_dwa = cfg["dwa"]
    use_smooth = cfg["smooth"]
    risk_switch = cfg.get("risk_switch", True)
    euclid_only = cfg.get("euclid_only", False) or not cfg["adaptive"]
    astar_risk = cfg["astar_risk"] and not cfg.get("euclid_only", False)
    planner_kind = cfg.get("planner", "astar")

    env, uav = B.build_scenario(seed)
    smoother = B.A.PathSmoother()
    dwa = _make_dwa(cfg, uav, env)

    if not cfg["adaptive"] or cfg.get("euclid_only", False):
        planner_cls = B.RiskBlindAStar if not astar_risk else B.EuclideanAStar
    else:
        planner_cls = B.A.AdaptiveAStar

    def do_replan():
        if use_kf:
            env.update_kf_trackers(sim_time)
        env.compute_risk_grid(uav, sim_time, use_kf=use_kf, n_steps=PRED_STEPS)
        planner = planner_cls(env, uav, B.GOAL, risk_threshold=B.ASTAR_RISK_THRESHOLD, k=B.BETA)
        raw = planner.run()
        if use_smooth:
            out, _n, _cap = B.fixes.safe_smooth(
                raw, env, uav.radius, B.A.PathSmoother,
                risk_threshold=B.SMOOTH_RISK_THRESHOLD, num_points=150)
        else:
            out = raw
        return out

    sim_time = 0.0
    env.compute_risk_grid(uav, 0.0, use_kf=use_kf, n_steps=PRED_STEPS)
    path = do_replan()

    last_replan = 0.0
    dwa_active = False
    dwa_start = 0.0
    consecutive_high = 0

    traj = [(float(uav.pos[0]), float(uav.pos[1]))]
    snapshots = {}
    snap_times = sorted(snapshot_times)
    snap_ptr = 0

    def maybe_snap():
        nonlocal snap_ptr
        while snap_ptr < len(snap_times) and sim_time + 1e-9 >= snap_times[snap_ptr]:
            snapshots[snap_times[snap_ptr]] = env.eval_risk_grid.copy()
            snap_ptr += 1

    env.compute_eval_risk_grid(uav, n_steps=PRED_STEPS)
    maybe_snap()

    outcome = "timeout"
    for _ in range(int(B.MAX_SIM_TIME / B.DT)):
        sim_time += B.DT
        env.update_dynamic(B.DT)
        if use_kf:
            env.update_kf_trackers(sim_time)
        env.compute_risk_grid(uav, sim_time, use_kf=use_kf, n_steps=PRED_STEPS)
        env.compute_eval_risk_grid(uav, n_steps=PRED_STEPS)
        maybe_snap()

        if B.in_collision(env, uav):
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

    return dict(cfg=cfg_name, outcome=outcome, trajectory=traj,
                snapshots=snapshots, sim_time=sim_time)


def _draw_obstacles(ax, env, dynamic_initial=True):
    for (x, y, r) in B.STATIC_OBSTACLES:
        ax.add_patch(plt.Circle((x, y), r, fc=STATIC_FC, ec=STATIC_EC, lw=0.6, zorder=1))
    if dynamic_initial:
        for o in env.dynamic_obstacles:
            ax.add_patch(plt.Circle((o.x, o.y), o.radius, fc=DYN_FC, ec=DYN_EC, lw=0.6, zorder=1))


def _style_axis(ax, xlim=(0, 200), ylim=(0, 200)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.tick_params(color=MUTED, labelcolor=INK)
    for s in ax.spines.values():
        s.set_color(MUTED)


def fig_trajectory_overlay(seed=0):
    """figure7.png -- trajectory comparison of the three key methods."""
    runs = {c: run_record(c, seed) for c in ["astar_dwa", "proposed", "proposed_pred_dwa"]}
    env, _ = B.build_scenario(seed)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _draw_obstacles(ax, env, dynamic_initial=True)

    for i, c in enumerate(["astar_dwa", "proposed", "proposed_pred_dwa"]):
        xs = [p[0] for p in runs[c]["trajectory"]]
        ys = [p[1] for p in runs[c]["trajectory"]]
        lw = 2.4 if c == "proposed_pred_dwa" else 1.4
        ax.plot(xs, ys, color=CAT[i], lw=lw, label=CONFIG_LABELS[c], zorder=3, alpha=0.95)
        if runs[c]["outcome"] == "collision":
            ax.plot(xs[-1], ys[-1], marker="x", color=CAT[i], ms=10, mew=2.2, zorder=4)

    ax.plot(B.START[0], B.START[1], marker="s", color=START_COLOR, ms=11,
            label="Start", zorder=5)
    ax.plot(B.GOAL[0], B.GOAL[1], marker="*", color=GOAL_COLOR, ms=17,
            label="Goal", zorder=5)

    _style_axis(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Trajectory comparison (seed %d)" % seed, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig("figure7.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figure7.png  outcomes=", {CONFIG_LABELS[c]: runs[c]["outcome"] for c in runs})


def fig_simulation_results(seed=0):
    """figure6.png -- a representative successful run of the complete framework."""
    rec = run_record("proposed_pred_dwa", seed)
    env, _ = B.build_scenario(seed)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _draw_obstacles(ax, env, dynamic_initial=True)

    xs = [p[0] for p in rec["trajectory"]]
    ys = [p[1] for p in rec["trajectory"]]
    ax.plot(xs, ys, color=CAT[2], lw=2.0, zorder=3, label="UAV trajectory")

    ax.plot(B.START[0], B.START[1], marker="s", color=START_COLOR, ms=11,
            label="Start", zorder=5)
    ax.plot(B.GOAL[0], B.GOAL[1], marker="*", color=GOAL_COLOR, ms=17,
            label="Goal", zorder=5)

    _style_axis(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Representative run of the complete framework (outcome: %s)"
                 % rec["outcome"], fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig("figure6.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figure6.png  outcome=", rec["outcome"])


def fig_risk_evolution(seed=0, times=(0, 30, 60, 90)):
    """figure5.png -- risk-field snapshots during a proposed run."""
    res = run_record("proposed", seed, snapshot_times=times)
    env, _ = B.build_scenario(seed)

    cmap = LinearSegmentedColormap.from_list(
        "risk_blue",
        ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
         "#184f95", "#0d366b"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    vmax = 1.0
    for ax, t in zip(axes.ravel(), times):
        grid = res["snapshots"].get(t)
        if grid is None:
            continue
        im = ax.imshow(grid.T, origin="lower", extent=[0, 200, 0, 200],
                       cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
        for o in env.dynamic_obstacles:
            ax.add_patch(plt.Circle((o.x, o.y), o.radius, fc="none", ec=DYN_EC,
                                    lw=0.8, zorder=3))
        if t == 0:
            ux, uy = B.START
        else:
            step = int(t / B.DT)
            pts = res["trajectory"]
            ux, uy = pts[min(step, len(pts) - 1)]
        ax.plot(ux, uy, marker="o", color=INK, ms=5, zorder=4)
        ax.set_title("t = %d s" % t, fontsize=10, color=INK)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_aspect("equal")
        ax.tick_params(color=MUTED, labelcolor=INK)
        for s in ax.spines.values():
            s.set_color(MUTED)

    fig.suptitle("Risk-field evolution during a proposed run (seed %d)" % seed,
                 fontsize=10, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    cbar = fig.colorbar(im, ax=axes, label="Risk value", fraction=0.046, pad=0.02)
    cbar.ax.tick_params(color=MUTED, labelcolor=INK)
    fig.savefig("figure5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figure5.png")


def fig_boxplots():
    """figure8.png -- box plots of the 100-run distributions (experiment_data_v4.xlsx)."""
    import openpyxl

    wb = openpyxl.load_workbook("experiment_data_v4.xlsx")
    ws = wb["Raw runs"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))

    def col(name):
        header = [c.value for c in ws[1]]
        return header.index(name)

    c_path = col("Path Length (m)")
    c_risk = col("Mean Local Risk")
    c_track = col("Tracking Error (m)")
    c_rep = col("Mean Replanning Time (s)")
    c_succ = col("Success")

    label_to_key = {v: k for k, v in CONFIG_LABELS.items()}
    # Curate the set of configurations shown (methods with meaningful data).
    shown = ["astar_dwa", "ia_only", "proposed", "proposed_pred_dwa"]
    data = {cfg: {"path": [], "risk": [], "track": [], "replan": []} for cfg in shown}
    for r in rows:
        cfg = label_to_key.get(r[0])
        if cfg is None or cfg not in shown:
            continue
        succ = r[c_succ]
        data[cfg]["replan"].append(r[c_rep])  # replan time over all runs
        if succ == 1:  # trajectory metrics over successful runs only
            data[cfg]["path"].append(r[c_path])
            data[cfg]["risk"].append(r[c_risk])
            data[cfg]["track"].append(r[c_track])

    panels = [
        ("Path length (m)", "path", "Path length (m)"),
        ("Tracking error (m)", "track", "Tracking error (m)"),
        ("Mean local risk", "risk", "Mean local risk value"),
        ("Replanning time (s)", "replan", "Replanning time (s)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6))
    box_color = "#2a78d6"

    for ax, (title, key, ylabel) in zip(axes.ravel(), panels):
        series = []
        labels = []
        for cfg in shown:
            vals = [v for v in data[cfg][key]
                    if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                series.append(vals)
                labels.append(CONFIG_LABELS[cfg])
        bp = ax.boxplot(series, labels=labels, patch_artist=True, widths=0.55,
                        medianprops=dict(color=INK, lw=1.4),
                        whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED),
                        flierprops=dict(marker="o", ms=3.5, mec=MUTED, mfc="none", alpha=0.6))
        for patch in bp["boxes"]:
            patch.set_facecolor(box_color)
            patch.set_edgecolor(INK)
            patch.set_linewidth(0.9)
            patch.set_alpha(0.75)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="x", labelrotation=15, labelsize=7.5, color=MUTED, labelcolor=INK)
        ax.tick_params(axis="y", labelsize=8, color=MUTED, labelcolor=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(MUTED)

    fig.suptitle("Distribution of metrics over 100 runs", fontsize=10, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("figure8.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figure8.png")


def fig_prediction(seed=0):
    """figure3.png -- Kalman-filter prediction of dynamic obstacles, with the
    growing uncertainty ellipse over the prediction horizon."""
    env, uav = B.build_scenario(seed)
    # advance the filter so the velocity estimate has converged
    for t in range(10):
        env.update_dynamic(B.DT)
        env.update_kf_trackers(t * B.DT)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    horizon = 7  # 1.4 s DWA prediction window
    shown = env.dynamic_obstacles[:3]
    for obs in shown:
        tracker = env.get_dynamic_tracker(obs)
        # current position
        ax.add_patch(plt.Circle((obs.x, obs.y), obs.radius, fc=DYN_FC, ec=DYN_EC,
                                lw=0.8, zorder=3))
        # predicted positions + covariance
        preds, covs = [], []
        for k in range(1, horizon + 1):
            ps, pc = tracker.get_predicted_state(k)
            preds.append((float(ps[0]), float(ps[1])))
            covs.append(pc[0:2, 0:2].copy())
        ax.plot([p[0] for p in preds], [p[1] for p in preds], '--', color=DYN_EC,
                lw=1.3, zorder=2)
        ax.plot(preds[-1][0], preds[-1][1], 'o', color=DYN_EC, ms=4, zorder=3)
        # 2-sigma uncertainty ellipse at the horizon
        cov = covs[-1]
        w, v = np.linalg.eigh(cov)
        w = np.maximum(w, 1e-6)
        angle = math.degrees(math.atan2(v[1, 0], v[0, 0]))
        ax.add_patch(Ellipse(preds[-1], 2 * math.sqrt(w[0]), 2 * math.sqrt(w[1]),
                             angle=angle, fill=False, ec=DYN_EC, lw=0.9,
                             linestyle=':', zorder=4))
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Kalman-filter prediction of dynamic obstacles", fontsize=10, color=INK)
    ax.tick_params(color=MUTED, labelcolor=INK)
    for s in ax.spines.values():
        s.set_color(MUTED)
    # legend
    ax.plot([], [], 'o', color=DYN_EC, ms=7, label="Current position")
    ax.plot([], [], '--', color=DYN_EC, lw=1.3, label="Predicted trajectory")
    ax.plot([], [], ':', color=DYN_EC, lw=1.0, label="2-sigma uncertainty")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figure3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figure3.png")


def fig_risk_field(seed=0):
    """figure4.png -- the forward-looking risk field (Gaussian contributions at
    predicted obstacle positions, spreading with prediction uncertainty)."""
    env, uav = B.build_scenario(seed)
    for t in range(10):
        env.update_dynamic(B.DT)
        env.update_kf_trackers(t * B.DT)
    env.compute_risk_grid(uav, 1.0, use_kf=True, n_steps=PRED_STEPS)

    cmap = LinearSegmentedColormap.from_list(
        "risk_blue",
        ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
         "#184f95", "#0d366b"])

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    grid = env.risk_grid
    im = ax.imshow(grid.T, origin="lower", extent=[0, 200, 0, 200],
                   cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal", zorder=1)
    # static obstacles
    for (x, y, r) in B.STATIC_OBSTACLES:
        ax.add_patch(plt.Circle((x, y), r, fc="none", ec=STATIC_EC, lw=0.7, zorder=2))
    # dynamic obstacles (current positions)
    for o in env.dynamic_obstacles:
        ax.add_patch(plt.Circle((o.x, o.y), o.radius, fc="none", ec=DYN_EC, lw=0.9, zorder=3))
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Forward-looking risk field", fontsize=10, color=INK)
    ax.tick_params(color=MUTED, labelcolor=INK)
    for s in ax.spines.values():
        s.set_color(MUTED)
    cbar = fig.colorbar(im, ax=ax, label="Risk value", fraction=0.046, pad=0.02)
    cbar.ax.tick_params(color=MUTED, labelcolor=INK)
    fig.tight_layout()
    fig.savefig("figure4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figure4.png")


if __name__ == "__main__":
    fig_prediction(seed=0)
    fig_risk_field(seed=0)
    fig_simulation_results(seed=0)
    fig_trajectory_overlay(seed=1)   # seed 1: A*+DWA collides, complete framework succeeds
    fig_risk_evolution(seed=0)
    fig_boxplots()
