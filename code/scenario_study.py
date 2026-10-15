"""
scenario_study.py -- Additional experiments requested by Reviewers 2 and 3 that
the nominal 12-static / 10-dynamic scene did not cover:

  R2 #24 (complex dynamic motion):     crossing / head-on obstacle trajectories
  R2 #35 (scalability / density):      sweep over the number of dynamic obstacles
                                       (+ a crowded-static scene)
  R2 #36 / R3 #42 (structured scenes): narrow passage, bottleneck, dead-end

These run the SAME paired-seed methodology and the SAME metric definitions as
benchmark.py (Section 5.2): identical seed -> identical scene for every
configuration, collision at uav.radius + obs.radius + 0.5 m, success-rate Wilson
95% CI, metrics over successful runs only. The only thing that changes between
studies is the environment geometry / obstacle motion, so the results are
directly comparable with Tables 1-2 of the manuscript.

Configurations (the five-point ladder of the paper, enough to isolate the
prediction/planning/avoidance layers without the full 10):
    astar             Traditional A*        (floor)
    ia_only           IA* w/o DWA           (planning layer only)
    astar_dwa         A* + DWA              (state-of-the-art baseline)
    proposed          full framework, static-obstacle DWA
    proposed_pred_dwa full framework + prediction-aware DWA   (the method)

Run:
    py scenario_study.py --study density    --runs 100 --jobs 7
    py scenario_study.py --study structured --runs 100 --jobs 7
    py scenario_study.py --study crossing   --runs 100 --jobs 7
    py scenario_study.py --study all --runs 100 --jobs 7

Outputs one xlsx per study under ../data/ (raw rows + a per-scenario summary).
"""

import argparse
import math
import os
import platform
import sys
import time

import numpy as np

import benchmark as B

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

CONFIGS = ["astar", "ia_only", "astar_dwa", "proposed", "proposed_pred_dwa"]

# Nominal obstacle sets, taken from benchmark.py.
NOM_STATIC = list(B.STATIC_OBSTACLES)                                  # (x, y, r)
NOM_DYN = [(r, vx, vy) for (_x, _y, r, vx, vy) in B.DYNAMIC_OBSTACLES]  # (r, vx, vy)


# --------------------------------------------------------------------------
# Deterministic obstacle generation (fixed seeds -> reproducible layouts).
# Only the *initial positions* of dynamic obstacles are re-randomised per run
# seed (Section 5.2 methodology); radii/velocities are held fixed.
# --------------------------------------------------------------------------
def gen_dyn_specs(n, seed):
    """n extra dynamic-obstacle (radius, vx, vy) specs, speeds ~0.8-1.4 m/s."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r = float(rng.uniform(3.0, 5.0))
        a = float(rng.uniform(0.0, 2.0 * math.pi))
        s = float(rng.uniform(0.8, 1.4))
        out.append((r, s * math.cos(a), s * math.sin(a)))
    return out


def gen_static_specs(n, seed, existing):
    """n extra static (x, y, r) circles clear of start/goal and of each other."""
    rng = np.random.default_rng(seed)
    out = []
    attempts = 0
    while len(out) < n and attempts < 10000:
        attempts += 1
        x = float(rng.uniform(8.0, 192.0))
        y = float(rng.uniform(8.0, 192.0))
        r = float(rng.uniform(4.0, 7.0))
        if math.hypot(x - B.START[0], y - B.START[1]) < 30.0:
            continue
        if math.hypot(x - B.GOAL[0], y - B.GOAL[1]) < 30.0:
            continue
        if any(math.hypot(x - ex[0], y - ex[1]) < r + ex[2] + 4.0
               for ex in existing + out):
            continue
        out.append((x, y, r))
    return out


# --------------------------------------------------------------------------
# Wall construction. Walls are rows of overlapping circles so they stay inside
# the existing circular-obstacle model (A*, DWA and the collision metric all
# already handle circles).
# --------------------------------------------------------------------------
def add_wall(env, x0, y0, x1, y1, r=1.5, spacing=2.0):
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(round(dist / spacing)))
    for i in range(n + 1):
        t = i / n
        env.add_static_obstacle(x0 + t * (x1 - x0), y0 + t * (y1 - y0), r)


def place_dynamic(env, rng, radius):
    """Random initial position clear of every static obstacle and of start/goal."""
    for _ in range(2000):
        x = float(rng.uniform(env.bounds[0] + radius, env.bounds[1] - radius))
        y = float(rng.uniform(env.bounds[2] + radius, env.bounds[3] - radius))
        if any(math.hypot(x - o.x, y - o.y) < radius + o.radius + 2.0
               for o in env.static_obstacles):
            continue
        if math.hypot(x - B.START[0], y - B.START[1]) < 25.0:
            continue
        if math.hypot(x - B.GOAL[0], y - B.GOAL[1]) < 25.0:
            continue
        return x, y
    return float(rng.uniform(20.0, 180.0)), float(rng.uniform(20.0, 180.0))


# --------------------------------------------------------------------------
# Scenario specs. Each is a dict consumed by build_spec_scenario().
# --------------------------------------------------------------------------
def density_specs():
    specs = {}
    specs["dyn5"] = dict(kind="density", name="5 dynamic (12 static)",
                         static=NOM_STATIC, dyn=NOM_DYN[:5], walls=[])
    specs["dyn10"] = dict(kind="density", name="10 dynamic (12 static)",
                          static=NOM_STATIC, dyn=NOM_DYN, walls=[])
    specs["dyn15"] = dict(kind="density", name="15 dynamic (12 static)",
                          static=NOM_STATIC, dyn=NOM_DYN + gen_dyn_specs(5, 101), walls=[])
    specs["dyn20"] = dict(kind="density", name="20 dynamic (12 static)",
                          static=NOM_STATIC, dyn=NOM_DYN + gen_dyn_specs(10, 102), walls=[])
    crowded = NOM_STATIC + gen_static_specs(12, 201, NOM_STATIC)
    specs["static24"] = dict(kind="density", name="10 dynamic (24 static)",
                             static=crowded, dyn=NOM_DYN, walls=[])
    return specs


def structured_specs():
    return {
        # A full-spanning vertical wall with a 16 m gap -> a genuine narrow passage.
        "narrow": dict(
            kind="structured", name="Narrow passage",
            static=[], dyn=NOM_DYN[:8],
            walls=[(100.0, 0.0, 100.0, 92.0, 1.5),
                   (100.0, 108.0, 100.0, 200.0, 1.5)]),
        # A full-spanning horizontal barrier with a 20 m central gap -> bottleneck.
        "bottleneck": dict(
            kind="structured", name="Bottleneck",
            static=[], dyn=NOM_DYN[:8],
            walls=[(0.0, 100.0, 90.0, 100.0, 1.5),
                   (110.0, 100.0, 200.0, 100.0, 1.5)]),
        # A concave U-shape sitting on the direct start->goal line -> dead-end trap.
        "deadend": dict(
            kind="structured", name="Dead-end (concave U)",
            static=[], dyn=NOM_DYN[:8],
            walls=[(70.0, 70.0, 130.0, 70.0, 1.5),
                   (70.0, 70.0, 70.0, 130.0, 1.5),
                   (130.0, 70.0, 130.0, 130.0, 1.5)]),
    }


def crossing_specs():
    """Straight horizontal/vertical movers that repeatedly cross the diagonal
    start->goal route (crossing / head-on encounters). Constant velocity, so the
    Kalman constant-velocity model remains a *valid* predictor -- the realistic
    'other aircraft flying straight' case, complementing the manoeuvre study."""
    dyn = [
        (3.5, 1.30, 0.00), (3.5, -1.30, 0.00), (4.0, 0.00, 1.20), (4.0, 0.00, -1.20),
        (3.0, 1.10, 0.00), (3.0, -1.10, 0.00), (4.5, 0.00, 1.00), (4.5, 0.00, -1.00),
        (3.5, 1.00, 0.00), (3.5, 0.00, 1.00),
    ]
    return {"crossing": dict(kind="crossing", name="Crossing / head-on traffic",
                             static=NOM_STATIC, dyn=dyn, walls=[])}


def build_spec_scenario(spec, seed):
    rng = np.random.default_rng(seed)
    env = B.BenchEnvironment(grid_size=B.GRID_SIZE, resolution=B.RESOLUTION,
                             bounds=B.BOUNDS)
    for (x, y, r) in spec["static"]:
        env.add_static_obstacle(x, y, r)
    for wall in spec["walls"]:
        add_wall(env, *wall)
    for (radius, vx, vy) in spec["dyn"]:
        x, y = place_dynamic(env, rng, radius)
        env.add_dynamic_obstacle(x, y, radius, vx, vy)

    uav = B.A.UAV(pos=B.START, **B.UAV_KW)
    uav.heading = math.atan2(B.GOAL[1] - B.START[1], B.GOAL[0] - B.START[0])
    return env, uav


# --------------------------------------------------------------------------
# Worker plumbing (spawn-safe on Windows: the scenario is re-installed in each
# worker via the multiprocessing initializer).
# --------------------------------------------------------------------------
def _init(spec):
    B.START = (10.0, 10.0)
    B.GOAL = (180.0, 180.0)
    B.BOUNDS = (0, 200, 0, 200)
    B.GRID_SIZE = (100, 100)
    B.RESOLUTION = 2.0
    B.STATIC_OBSTACLES = spec["static"]
    B.build_scenario = lambda seed, maneuver=False: build_spec_scenario(spec, seed)


def _worker(job):
    cfg, seed = job
    try:
        return B.run_episode(cfg, seed, B.N_PRED_STEPS, maneuver=False)
    except Exception as exc:
        return dict(config=cfg, seed=seed, success=0, outcome="error:%s" % exc,
                    path_length=float("nan"), mean_risk=float("nan"),
                    max_risk=float("nan"), tracking_error=float("nan"),
                    replan_time=float("nan"), n_replans=0, sim_time=0.0,
                    full_step_time=float("nan"))


def run_study(specs, runs, jobs, out_path):
    import multiprocessing as mp

    t0 = time.time()
    all_rows = []
    summaries = []

    for name, spec in specs.items():
        jobs_list = [(c, s) for c in CONFIGS for s in range(runs)]
        rows = []
        with mp.Pool(jobs, initializer=_init, initargs=(spec,)) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, jobs_list), 1):
                rows.append(r)
                if i % 100 == 0 or i == len(jobs_list):
                    print("  [%s] %3d/%d  (%.0f s elapsed)"
                          % (name, i, len(jobs_list), time.time() - t0), flush=True)

        summary = _summarise(name, spec["name"], rows)
        summaries.append(summary)
        for r in rows:
            r["scenario"] = spec["name"]
        all_rows.extend(rows)
        _print_scenario(summary)

    _write_xlsx(summaries, all_rows, out_path)
    print("\nWrote %s (total %.0f s)" % (out_path, time.time() - t0))


def _summarise(name, label, rows):
    out = dict(name=name, label=label)
    for cfg in CONFIGS:
        rs = [r for r in rows if r["config"] == cfg]
        n = len(rs)
        succ = [r for r in rs if r["success"]]
        lo, hi = B.wilson_ci(len(succ), n) if n else (float("nan"), float("nan"))

        def m(key, src):
            vals = [r[key] for r in src if not math.isnan(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        out[cfg] = dict(
            runs=n,
            success=100.0 * len(succ) / n if n else float("nan"),
            lo=100.0 * lo, hi=100.0 * hi,
            path=m("path_length", succ),
            track=m("tracking_error", succ),
            full_step=m("full_step_time", rs),
            n_collision=sum(1 for r in rs if r["outcome"] == "collision"),
            n_timeout=sum(1 for r in rs if r["outcome"] == "timeout"),
            n_stuck=sum(1 for r in rs if r["outcome"] == "stuck"),
        )
    return out


def _print_scenario(s):
    print("\n=== %s: %s ===" % (s["name"], s["label"]))
    print("%-22s %7s %-16s %9s %9s %10s %s" % (
        "Configuration", "Succ%", "95% CI", "Path(m)", "TrkErr", "FullStep(s)", "Fail(C/T/S)"))
    for cfg in CONFIGS:
        c = s[cfg]
        ci = "[%4.1f,%4.1f]" % (c["lo"], c["hi"])
        fail = "%d/%d/%d" % (c["n_collision"], c["n_timeout"], c["n_stuck"])
        print("%-22s %7.1f %-16s %9.1f %9.2f %10.4f %s" % (
            B.CONFIGS[cfg]["label"], c["success"], ci, c["path"], c["track"],
            c["full_step"], fail))


def _write_xlsx(summaries, rows, path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment

    wb = Workbook()
    bold = Font(bold=True)

    ws = wb.active
    ws.title = "Summary"
    ws.append(["Scenario", "Configuration", "Runs", "Success Rate (%)",
               "95% CI Low", "95% CI High", "Path Length (m)",
               "Tracking Error (m)", "Full Step Time (s)",
               "Collisions", "Timeouts", "Stuck"])
    for c in ws[1]:
        c.font = bold
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for s in summaries:
        for cfg in CONFIGS:
            c = s[cfg]
            ws.append([s["label"], B.CONFIGS[cfg]["label"], c["runs"],
                       round(c["success"], 1), round(c["lo"], 1), round(c["hi"], 1),
                       round(c["path"], 1), round(c["track"], 2),
                       round(c["full_step"], 4), c["n_collision"], c["n_timeout"],
                       c["n_stuck"]])
    for col in "ABCDEFGHIJKL":
        ws.column_dimensions[col].width = 18

    ws2 = wb.create_sheet("Raw runs")
    ws2.append(["Scenario", "Configuration", "Seed", "Success", "Outcome",
                "Path Length (m)", "Tracking Error (m)", "Mean Replan Time (s)",
                "Sim Time (s)", "Full Step Time (s)"])
    for c in ws2[1]:
        c.font = bold
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for r in sorted(rows, key=lambda r: (r["scenario"], r["config"], r["seed"])):
        ws2.append([r.get("scenario", ""), B.CONFIGS[r["config"]]["label"],
                    r["seed"], r["success"], r["outcome"],
                    round(r["path_length"], 2), round(r["tracking_error"], 2),
                    round(r["replan_time"], 4), round(r["sim_time"], 1),
                    round(r["full_step_time"], 4)])
    ws2.column_dimensions["A"].width = 20
    ws2.column_dimensions["B"].width = 22
    for col in "CDEFGHIJ":
        ws2.column_dimensions[col].width = 14
    ws2.freeze_panes = "A2"

    wb.save(path)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="all",
                    choices=["density", "structured", "crossing", "all"])
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out-dir", default=DATA_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("Scenario study: %s | %d seeds x %d configs | %d workers\n"
          % (args.study, args.runs, len(CONFIGS), args.jobs))

    studies = []
    if args.study in ("density", "all"):
        studies.append(("density", density_specs(), "experiment_data_density.xlsx"))
    if args.study in ("structured", "all"):
        studies.append(("structured", structured_specs(), "experiment_data_structured.xlsx"))
    if args.study in ("crossing", "all"):
        studies.append(("crossing", crossing_specs(), "experiment_data_crossing.xlsx"))

    for study, specs, fname in studies:
        print("\n########## %s ##########" % study)
        run_study(specs, args.runs, args.jobs, os.path.join(args.out_dir, fname))

    print("\nHardware: %s | %d cores | Python %s"
          % (platform.platform(), os.cpu_count() or 1, platform.python_version()))


if __name__ == "__main__":
    main()
