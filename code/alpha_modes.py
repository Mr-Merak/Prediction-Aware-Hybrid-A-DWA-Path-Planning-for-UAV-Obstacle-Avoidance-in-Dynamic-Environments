"""
alpha_modes.py -- sigmoid vs linear vs threshold risk-adaptive weight.

Review 2, round 2, comment #2: the ablation does not compare the sigmoid
weight of Eq. (16) against the two natural alternatives (a linear blend and a
hard threshold).  This script runs the same 100 paired seeds for all three
shapes, in the proposed configuration and in the complete framework.

Modes (benchmark.ALPHA_MODE):
    sigmoid   : alpha = 1 / (1 + exp(-beta (rbar - r0)))      (manuscript Eq. 16)
    linear    : alpha = clip(rbar, 0, 1)                       (linear blend)
    threshold : alpha = 1 if rbar > r0 else 0                  (hard switch)

Run:
    py alpha_modes.py --runs 100 --jobs 7
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

MODES = ["sigmoid", "linear", "threshold"]
CONFIGS = ["proposed", "proposed_pred_dwa"]


def _init(mode):
    B.ALPHA_MODE = mode
    B.PREDICTOR = "cv"


def _worker(job):
    cfg, seed = job
    try:
        return B.run_episode(cfg, seed, B.N_PRED_STEPS, maneuver=False)
    except Exception as exc:
        return dict(config=cfg, seed=seed, success=0, outcome="error:%s" % exc,
                    path_length=float("nan"), tracking_error=float("nan"),
                    full_step_time=float("nan"), replan_time=float("nan"))


def _summarise(rows):
    out = {}
    for cfg in CONFIGS:
        rs = [r for r in rows if r["config"] == cfg]
        n = len(rs)
        succ = [r for r in rs if r["success"]]
        lo, hi = B.wilson_ci(len(succ), n) if n else (float("nan"), float("nan"))

        def m(key, src):
            vals = [r[key] for r in src if not math.isnan(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        out[cfg] = dict(n=n, succ=len(succ), rate=100.0 * len(succ) / n if n else float("nan"),
                        lo=100.0 * lo, hi=100.0 * hi,
                        path=m("path_length", succ), track=m("tracking_error", succ),
                        step=m("full_step_time", rs),
                        collision=sum(1 for r in rs if r["outcome"] == "collision"),
                        timeout=sum(1 for r in rs if r["outcome"] == "timeout"),
                        stuck=sum(1 for r in rs if r["outcome"] == "stuck"))
    return out


def _mcnemar(a, b):
    from scipy import stats as st
    b_cnt = int(np.sum((a == 1) & (b == 0)))
    c_cnt = int(np.sum((a == 0) & (b == 1)))
    p = st.binomtest(min(b_cnt, c_cnt), b_cnt + c_cnt, 0.5).pvalue if (b_cnt + c_cnt) else float("nan")
    return b_cnt, c_cnt, p


def _print_table(rows_by_mode):
    print("\n===== alpha(rbar) shape comparison (100 paired seeds) =====")
    print("%-10s %-22s %7s %-16s %9s %9s %8s" % (
        "Mode", "Configuration", "Succ%", "95% CI", "Path(m)", "TrkErr", "Step(s)"))
    for mode in MODES:
        for cfg in CONFIGS:
            s = rows_by_mode[mode][cfg]
            print("%-10s %-22s %6.1f  [%4.1f,%4.1f] %9.1f %9.2f %8.4f" % (
                mode, B.CONFIGS[cfg]["label"], s["rate"], s["lo"], s["hi"],
                s["path"], s["track"], s["step"]))

    print("\nMcNemar vs sigmoid (paired seeds):")
    for cfg in CONFIGS:
        base = np.array([int(r["success"]) for r in
                         sorted(rows_by_mode["sigmoid"][cfg]["rows"], key=lambda x: x["seed"])])
        for mode in ["linear", "threshold"]:
            alt = np.array([int(r["success"]) for r in
                            sorted(rows_by_mode[mode][cfg]["rows"], key=lambda x: x["seed"])])
            b_cnt, c_cnt, p = _mcnemar(alt, base)
            print("  %-22s %-9s vs sigmoid: %+5.1f pp  (b/c=%d/%d, p=%.3f)" % (
                B.CONFIGS[cfg]["label"], mode,
                rows_by_mode[mode][cfg]["rate"] - rows_by_mode["sigmoid"][cfg]["rate"],
                b_cnt, c_cnt, p))


def _write_xlsx(rows_by_mode, path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment

    wb = Workbook()
    ws = wb.active
    ws.title = "Alpha shapes"
    ws.append(["Mode", "Configuration", "Runs", "Successes", "Success Rate (%)",
               "95% CI Low", "95% CI High", "Path Length (m)", "Tracking Error (m)",
               "Full Step Time (s)", "Collisions", "Timeouts", "Stuck"])
    for c in ws[1]:
        c.font = Font(bold=True)
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for mode in MODES:
        for cfg in CONFIGS:
            s = rows_by_mode[mode][cfg]
            ws.append([mode, B.CONFIGS[cfg]["label"], s["n"], s["succ"],
                       round(s["rate"], 1), round(s["lo"], 1), round(s["hi"], 1),
                       round(s["path"], 1) if not math.isnan(s["path"]) else None,
                       round(s["track"], 2) if not math.isnan(s["track"]) else None,
                       round(s["step"], 4) if not math.isnan(s["step"]) else None,
                       s["collision"], s["timeout"], s["stuck"]])
    for col in "ABCDEFGHIJKLM":
        ws.column_dimensions[col].width = 20

    ws2 = wb.create_sheet("Paired tests")
    ws2.append(["Configuration", "Comparison", "Delta (pp)", "Discordant (b/c)",
                "McNemar p"])
    for c in ws2[1]:
        c.font = Font(bold=True)
    for cfg in CONFIGS:
        base = np.array([int(r["success"]) for r in
                         sorted(rows_by_mode["sigmoid"][cfg]["rows"], key=lambda x: x["seed"])])
        for mode in ["linear", "threshold"]:
            alt = np.array([int(r["success"]) for r in
                            sorted(rows_by_mode[mode][cfg]["rows"], key=lambda x: x["seed"])])
            b_cnt, c_cnt, p = _mcnemar(alt, base)
            ws2.append([B.CONFIGS[cfg]["label"], "%s vs sigmoid" % mode,
                        round(rows_by_mode[mode][cfg]["rate"]
                              - rows_by_mode["sigmoid"][cfg]["rate"], 1),
                        "%d/%d" % (b_cnt, c_cnt), round(p, 4)])
    for col in "ABCDE":
        ws2.column_dimensions[col].width = 24

    # raw per-seed rows for reproducibility
    ws3 = wb.create_sheet("Raw runs")
    ws3.append(["Mode", "Configuration", "Seed", "Success", "Outcome",
                "Path Length (m)", "Tracking Error (m)", "Full Step Time (s)"])
    for mode in MODES:
        for cfg in CONFIGS:
            for r in sorted(rows_by_mode[mode][cfg]["rows"], key=lambda x: x["seed"]):
                ws3.append([mode, B.CONFIGS[cfg]["label"], r["seed"], r["success"],
                            r["outcome"],
                            None if math.isnan(r["path_length"]) else round(r["path_length"], 2),
                            None if math.isnan(r["tracking_error"]) else round(r["tracking_error"], 2),
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
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "experiment_data_alpha_modes.xlsx"))
    args = ap.parse_args()

    print("alpha(rbar) comparison: %d modes x %d configs x %d seeds | %d workers"
          % (len(MODES), len(CONFIGS), args.runs, args.jobs))

    rows_by_mode = {}
    t0 = time.time()
    for mode in MODES:
        print("\n########## %s ##########" % mode)
        jobs = [(cfg, seed) for cfg in CONFIGS for seed in range(args.runs)]
        with mp.Pool(args.jobs, initializer=_init, initargs=(mode,)) as pool:
            rows = [r for r in pool.imap_unordered(_worker, jobs)]
        rows_by_mode[mode] = _summarise(rows)
        for cfg in CONFIGS:
            rows_by_mode[mode][cfg]["rows"] = [r for r in rows if r["config"] == cfg]
        print("  elapsed %.0f s" % (time.time() - t0))

    _print_table(rows_by_mode)
    _write_xlsx(rows_by_mode, args.out)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
