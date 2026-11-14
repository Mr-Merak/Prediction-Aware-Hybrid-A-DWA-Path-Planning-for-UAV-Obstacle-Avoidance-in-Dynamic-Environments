"""
make_figure_3d6dof.py -- figure for the 6-DOF 3D validation.

Default mode ("bar") reads data/experiment_data_3d6dof.xlsx and draws a
two-panel success-rate bar chart with Wilson 95% confidence intervals for the
altitude-window and 3D-crossing scenarios.  Mode "traj" reads the saved
representative trajectories instead.

Run:
    py make_figure_3d6dof.py                 # bar chart -> figure3d_6dof.png
    py make_figure_3d6dof.py --mode traj     # trajectories -> figure3d_6dof_traj.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

COLORS = {
    "3D A* (Euclidean, pure pursuit)": "#7f7f7f",
    "3D IA* (risk-adaptive, pure pursuit)": "#1f77b4",
    "3D IA* + DWA": "#ff7f0e",
    "3D IA* + prediction-aware DWA": "#2ca02c",
}
SHORT = {
    "3D A* (Euclidean, pure pursuit)": "3D A*\n(Euclidean)",
    "3D IA* (risk-adaptive, pure pursuit)": "3D IA*\n(risk-adaptive)",
    "3D IA* + DWA": "3D IA*\n+ DWA",
    "3D IA* + prediction-aware DWA": "3D IA*\n+ pred-DWA",
}
SCENARIOS = ["Altitude window (3D movers)", "3D crossing traffic"]
TITLES = {"Altitude window (3D movers)": "Altitude window (3D movers)",
          "3D crossing traffic": "3D crossing traffic"}


def bar_chart(xlsx, out):
    from openpyxl import load_workbook
    wb = load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb["Summary"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, sc in zip(axes, SCENARIOS):
        rs = [r for r in rows if r[0] == sc]
        labels = [SHORT[r[1]] for r in rs]
        vals = [r[4] for r in rs]
        los = [r[4] - r[5] for r in rs]
        his = [r[6] - r[4] for r in rs]
        colors = [COLORS[r[1]] for r in rs]
        x = list(range(len(rs)))
        ax.bar(x, vals, yerr=[los, his], capsize=4, color=colors, alpha=0.85,
               edgecolor="black", linewidth=0.6)
        for xi, v, hi in zip(x, vals, his):
            ax.text(xi, min(v + hi + 1.5, 106), "%d" % v, ha="center",
                    va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Success rate (%)")
        ax.set_title(TITLES[sc])
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("Wrote %s" % out)


def traj_chart(json_path, out):
    with open(json_path) as f:
        traj = json.load(f)
    fig = plt.figure(figsize=(12, 5))
    for i, sc in enumerate(["window", "crossing3d"], 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        for cfg, color in [("astar3d", COLORS["3D A* (Euclidean, pure pursuit)"]),
                           ("ia3d", COLORS["3D IA* (risk-adaptive, pure pursuit)"]),
                           ("ia3d_dwa", COLORS["3D IA* + DWA"]),
                           ("ia3d_pred_dwa", COLORS["3D IA* + prediction-aware DWA"])]:
            key = "%s/%s" % (sc, cfg)
            if key not in traj:
                continue
            pts = traj[key]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts],
                    color=color, lw=1.6)
        ax.scatter([5], [50], [15], marker="o", s=40, c="k")
        ax.scatter([95], [50], [15], marker="*", s=90, c="k")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_zlim(0, 40)
        ax.set_box_aspect((100, 100, 40))
        ax.set_title(TITLES["Altitude window (3D movers)"] if sc == "window"
                     else TITLES["3D crossing traffic"])
        ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("Wrote %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bar", "traj"], default="bar")
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "bar":
        xlsx = args.input or os.path.join(ROOT, "data", "experiment_data_3d6dof.xlsx")
        out = args.out or os.path.join(ROOT, "figure3d_6dof.png")
        bar_chart(xlsx, out)
    else:
        js = args.input or os.path.join(ROOT, "data", "trajectories_3d6dof.json")
        out = args.out or os.path.join(ROOT, "figure3d_6dof_traj.png")
        traj_chart(js, out)


if __name__ == "__main__":
    main()
