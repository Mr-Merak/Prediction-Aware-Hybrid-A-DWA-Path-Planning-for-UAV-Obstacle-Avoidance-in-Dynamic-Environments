"""
sensitivity100.py -- parameter-sensitivity sweeps with 100 paired seeds.

Review 2, round 2, comment #7: the sensitivity study used 30 trials per point;
the reviewer asked for 100 trials (at least for beta and the prediction
horizon) or an explicit statement that the study is exploratory.  This script
re-runs the same four sweeps as Table 10 with 100 seeds and reports Wilson
95% confidence intervals, so the "no catastrophic degradation" claim is
supported rather than asserted.

Run:
    py sensitivity100.py --runs 100 --jobs 7
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

CONFIG = "proposed"          # same configuration as Table 10
NOM_BETA, NOM_PROC, NOM_OBS = 10.0, 0.5, 2.0

# (parameter label, list of values, kind)
SWEEPS = [
    ("prediction horizon (steps)", [5, 10, 15, 25, 40], "horizon"),
    ("beta (heuristic steepness)", [1.0, 5.0, 10.0, 20.0, 50.0], "beta"),
    ("observation noise R", [0.5, 2.0, 5.0, 10.0], "obs"),
    ("process noise Q", [0.1, 0.5, 2.0, 5.0], "proc"),
]


def _init(beta, proc, obs):
    B.BETA = beta
    B.KF_PROCESS_NOISE = proc
    B.KF_OBS_NOISE = obs
    B.PREDICTOR = "cv"


def _worker(seed_pred):
    seed, pred_steps = seed_pred
    try:
        return B.run_episode(CONFIG, seed, pred_steps, maneuver=False)
    except Exception as exc:
        return dict(config=CONFIG, seed=seed, success=0, outcome="error:%s" % exc,
                    path_length=float("nan"), mean_risk=float("nan"),
                    full_step_time=float("nan"))


def run_point(pred_steps, beta, proc, obs, runs, jobs):
    jobs_list = [(s, pred_steps) for s in range(runs)]
    with mp.Pool(jobs, initializer=_init, initargs=(beta, proc, obs)) as pool:
        rows = [r for r in pool.imap_unordered(_worker, jobs_list)]
    succ = sum(1 for r in rows if r["success"])
    lo, hi = B.wilson_ci(succ, runs)
    risks = [r["mean_risk"] for r in rows
             if r["success"] and not math.isnan(r["mean_risk"])]
    return dict(runs=runs, succ=succ, rate=100.0 * succ / runs,
                lo=100.0 * lo, hi=100.0 * hi,
                mean_risk=float(np.mean(risks)) if risks else float("nan"),
                collision=sum(1 for r in rows if r["outcome"] == "collision"),
                timeout=sum(1 for r in rows if r["outcome"] == "timeout"),
                stuck=sum(1 for r in rows if r["outcome"] == "stuck"),
                rows=rows)


def _write_xlsx(results, path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment

    wb = Workbook()
    ws = wb.active
    ws.title = "Sensitivity 100"
    ws.append(["Parameter", "Value", "Runs", "Successes", "Success Rate (%)",
               "95% CI Low", "95% CI High", "Mean Local Risk",
               "Collisions", "Timeouts", "Stuck"])
    for c in ws[1]:
        c.font = Font(bold=True)
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for label, value, s in results:
        ws.append([label, value, s["runs"], s["succ"], round(s["rate"], 1),
                   round(s["lo"], 1), round(s["hi"], 1),
                   round(s["mean_risk"], 4) if not math.isnan(s["mean_risk"]) else None,
                   s["collision"], s["timeout"], s["stuck"]])
    for col in "ABCDEFGHIJK":
        ws.column_dimensions[col].width = 20

    ws2 = wb.create_sheet("Raw runs")
    ws2.append(["Parameter", "Value", "Seed", "Success", "Outcome",
                "Path Length (m)", "Mean Local Risk", "Full Step Time (s)"])
    for label, value, s in results:
        for r in sorted(s["rows"], key=lambda x: x["seed"]):
            ws2.append([label, value, r["seed"], r["success"], r["outcome"],
                        None if math.isnan(r["path_length"]) else round(r["path_length"], 2),
                        None if math.isnan(r["mean_risk"]) else round(r["mean_risk"], 4),
                        None if math.isnan(r["full_step_time"]) else round(r["full_step_time"], 4)])
    wb.save(path)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "experiment_data_sensitivity100.xlsx"))
    args = ap.parse_args()

    print("sensitivity sweeps: %d seeds per point | %d workers" % (args.runs, args.jobs))
    results = []
    t0 = time.time()
    for label, values, kind in SWEEPS:
        print("\n--- %s ---" % label)
        for v in values:
            ps, beta, proc, obs = B.N_PRED_STEPS, NOM_BETA, NOM_PROC, NOM_OBS
            if kind == "horizon":
                ps = int(v)
            elif kind == "beta":
                beta = float(v)
            elif kind == "obs":
                obs = float(v)
            elif kind == "proc":
                proc = float(v)
            s = run_point(ps, beta, proc, obs, args.runs, args.jobs)
            results.append((label, v, s))
            print("  %-8s success=%5.1f%% [%4.1f,%4.1f]  (collisions=%d, stuck=%d)"
                  % (v, s["rate"], s["lo"], s["hi"], s["collision"], s["stuck"]),
                  flush=True)
            print("    elapsed %.0f s" % (time.time() - t0), flush=True)

    _write_xlsx(results, args.out)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
