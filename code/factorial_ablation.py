"""
factorial_ablation.py -- 2x2 factorial ablation (Reviewer 2, suggestion #2).

Factor A: risk-adaptive heuristic + Kalman risk field (global layer)  {off, on}
Factor B: prediction-aware DWA (local layer)                          {off, on}

Cells (benchmark.py configs):
    A off, B off = astar_dwa          "A* + DWA"
    A off, B on  = astar_pred_dwa     "A* + Pred-DWA"
    A on,  B off = proposed           "IA* + static DWA"
    A on,  B on  = proposed_pred_dwa  "IA* + Pred-DWA"   (the full method)

Runs on the nominal scene and the three structured scenes (narrow / bottleneck /
dead-end), paired by seed, and reports the 2x2 success-rate table, the simple main
effect of each factor within each level of the other, and the interaction
(difference of simple main effects) with a within-subjects permutation p-value.

Run:
    py factorial_ablation.py --runs 100 --jobs 7
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import benchmark as B
import scenario_study as SS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

CONFIGS = ["astar_dwa", "astar_pred_dwa", "proposed", "proposed_pred_dwa"]
# factor levels: (A_on, B_on)
CELL_KEYS = {
    "astar_dwa": (False, False),
    "astar_pred_dwa": (False, True),
    "proposed": (True, False),
    "proposed_pred_dwa": (True, True),
}


def _err(cfg, seed, exc):
    return dict(config=cfg, seed=seed, success=0, outcome="error:%s" % exc,
                path_length=float("nan"), mean_risk=float("nan"),
                max_risk=float("nan"), tracking_error=float("nan"),
                replan_time=float("nan"), n_replans=0, sim_time=0.0,
                full_step_time=float("nan"))


def _worker(job):
    """Module-level worker (spawn-safe on Windows)."""
    cfg, seed = job
    try:
        return B.run_episode(cfg, seed, B.N_PRED_STEPS, maneuver=False)
    except Exception as exc:
        return _err(cfg, seed, exc)


def _run_jobs(jobs_list, jobs, init_fn=None, init_args=None):
    """Run (cfg, seed) jobs through benchmark.run_episode, spawn-safe."""
    if init_fn is None:
        with mp.Pool(jobs) as pool:
            rows = [r for r in pool.imap_unordered(_worker, jobs_list)]
    else:
        with mp.Pool(jobs, initializer=init_fn, initargs=init_args) as pool:
            rows = [r for r in pool.imap_unordered(_worker, jobs_list)]
    return rows


def _init_structured(spec):
    B.START = (10.0, 10.0)
    B.GOAL = (180.0, 180.0)
    B.BOUNDS = (0, 200, 0, 200)
    B.GRID_SIZE = (100, 100)
    B.RESOLUTION = 2.0
    B.STATIC_OBSTACLES = spec["static"]
    B.build_scenario = lambda seed, maneuver=False: SS.build_spec_scenario(spec, seed)


def run_nominal(runs, jobs):
    jobs_list = [(c, s) for c in CONFIGS for s in range(runs)]
    t0 = time.time()
    rows = _run_jobs(jobs_list, jobs)
    return rows, time.time() - t0


def run_structured(name, spec, runs, jobs):
    jobs_list = [(c, s) for c in CONFIGS for s in range(runs)]
    t0 = time.time()
    rows = _run_jobs(jobs_list, jobs, _init_structured, (spec,))
    return rows, time.time() - t0


# --------------------------------------------------------------------------
# Factorial aggregation
# --------------------------------------------------------------------------
def factorial_table(rows):
    """Return dict with per-cell success stats + simple effects + interaction."""
    piv = {}
    for r in rows:
        piv.setdefault(r["config"], {})[r["seed"]] = r

    cells = {}
    for cfg in CONFIGS:
        rs = piv.get(cfg, {})
        n = len(rs)
        succ = sum(1 for r in rs.values() if r["success"])
        lo, hi = B.wilson_ci(succ, n) if n else (float("nan"), float("nan"))
        cells[cfg] = dict(n=n, succ=succ, rate=100.0 * succ / n if n else float("nan"),
                          lo=100.0 * lo, hi=100.0 * hi)

    # per-seed success bits, aligned
    seeds = sorted(set().union(*[set(piv.get(c, {})) for c in CONFIGS]))
    s00 = np.array([int(piv["astar_dwa"].get(s, {}).get("success", 0)) for s in seeds])
    s01 = np.array([int(piv["astar_pred_dwa"].get(s, {}).get("success", 0)) for s in seeds])
    s10 = np.array([int(piv["proposed"].get(s, {}).get("success", 0)) for s in seeds])
    s11 = np.array([int(piv["proposed_pred_dwa"].get(s, {}).get("success", 0)) for s in seeds])

    def mcnemar(a, b):
        b_cnt = int(np.sum((a == 1) & (b == 0)))
        c_cnt = int(np.sum((a == 0) & (b == 1)))
        from scipy import stats as st
        p = st.binomtest(min(b_cnt, c_cnt), b_cnt + c_cnt, 0.5).pvalue if (b_cnt + c_cnt) else float("nan")
        return b_cnt, c_cnt, p

    # simple main effect of B within each level of A
    b_at_a0 = cells["astar_pred_dwa"]["rate"] - cells["astar_dwa"]["rate"]
    b_at_a1 = cells["proposed_pred_dwa"]["rate"] - cells["proposed"]["rate"]
    # simple main effect of A within each level of B
    a_at_b0 = cells["proposed"]["rate"] - cells["astar_dwa"]["rate"]
    a_at_b1 = cells["proposed_pred_dwa"]["rate"] - cells["astar_pred_dwa"]["rate"]

    # McNemar tests for the simple effects (paired)
    b0_mc = mcnemar(s01, s00)   # B effect at A=0: astar_pred_dwa vs astar_dwa
    b1_mc = mcnemar(s11, s10)   # B effect at A=1: proposed_pred_dwa vs proposed
    a0_mc = mcnemar(s10, s00)   # A effect at B=0: proposed vs astar_dwa
    a1_mc = mcnemar(s11, s01)   # A effect at B=1: proposed_pred_dwa vs astar_pred_dwa

    # interaction: (B effect at A=1) - (B effect at A=0), per-seed contrast
    contrast = (s11 - s10) - (s01 - s00)   # in {-2..2}
    T_obs = float(np.mean(contrast))

    # within-subjects permutation: swap A rows (A=0 <-> A=1) per seed
    rng = np.random.default_rng(0)
    n_perm = 20000
    T_null = np.empty(n_perm)
    # precompute the two "swap" options per seed: keeping (B off, B on) pairs
    # original arrangement: (s00, s01) for A=0 row, (s10, s11) for A=1 row
    # swapped: A=0 row becomes (s10, s11), A=1 row becomes (s00, s01)
    # interaction contrast = (B at A1) - (B at A0) = (s11-s10) - (s01-s00)
    # after swap: A1 = s01, A0 = s11; contrast = (s01 - s00) - (s11 - s10) = -(original)
    for i in range(n_perm):
        flips = rng.integers(0, 2, size=len(seeds)).astype(bool)
        c = contrast.copy()
        c[flips] = -c[flips]
        T_null[i] = float(np.mean(c))
    p_interaction = float(np.mean(np.abs(T_null) >= abs(T_obs)))

    return dict(cells=cells, b_at_a0=b_at_a0, b_at_a1=b_at_a1,
                a_at_b0=a_at_b0, a_at_b1=a_at_b1,
                b0_mc=b0_mc, b1_mc=b1_mc, a0_mc=a0_mc, a1_mc=a1_mc,
                # interaction in percentage points: difference of the two simple
                # main effects of B (the per-seed contrast mean is the same quantity
                # in proportion units, used only for the permutation p-value).
                interaction=b_at_a1 - b_at_a0, interaction_p=p_interaction,
                n_seeds=len(seeds))


def print_table(name, label, ft):
    print("\n===== %s: %s =====" % (name, label))
    print("%-24s %-10s %-16s" % ("Configuration", "Succ%", "95% CI"))
    for cfg in CONFIGS:
        c = ft["cells"][cfg]
        print("  %-22s %6.1f   [%4.1f, %4.1f]  (%d/%d)"
              % (B.CONFIGS[cfg]["label"], c["rate"], c["lo"], c["hi"], c["succ"], c["n"]))
    print("  Simple main effect of Pred-DWA (B):")
    print("    at heuristic OFF: %+.1f pp (McNemar b/c=%d/%d, p=%.3f)"
          % (ft["b_at_a0"], ft["b0_mc"][0], ft["b0_mc"][1], ft["b0_mc"][2]))
    print("    at heuristic ON : %+.1f pp (McNemar b/c=%d/%d, p=%.3f)"
          % (ft["b_at_a1"], ft["b1_mc"][0], ft["b1_mc"][1], ft["b1_mc"][2]))
    print("  Simple main effect of adaptive heuristic (A):")
    print("    at Pred-DWA OFF: %+.1f pp (McNemar b/c=%d/%d, p=%.3f)"
          % (ft["a_at_b0"], ft["a0_mc"][0], ft["a0_mc"][1], ft["a0_mc"][2]))
    print("    at Pred-DWA ON : %+.1f pp (McNemar b/c=%d/%d, p=%.3f)"
          % (ft["a_at_b1"], ft["a1_mc"][0], ft["a1_mc"][1], ft["a1_mc"][2]))
    print("  Interaction (B effect at A=ON minus A=OFF): %+.1f pp, permutation p=%.4f"
          % (ft["interaction"], ft["interaction_p"]))


def write_xlsx(results, path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment
    wb = Workbook()
    bold = Font(bold=True)
    ws = wb.active
    ws.title = "2x2 factorial"
    ws.append(["Scenario", "Cell (A=heuristic, B=pred-DWA)", "Runs", "Successes",
               "Success Rate (%)", "95% CI Low", "95% CI High"])
    for c in ws[1]:
        c.font = bold
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    def cell_label(cfg):
        a, b = CELL_KEYS[cfg]
        return "heuristic %s / pred-DWA %s" % ("on" if a else "off", "on" if b else "off")

    for name, label, ft, _rows in results:
        for cfg in CONFIGS:
            c = ft["cells"][cfg]
            ws.append([label, cell_label(cfg), c["n"], c["succ"],
                       round(c["rate"], 1), round(c["lo"], 1), round(c["hi"], 1)])
    for col in "ABCDEFG":
        ws.column_dimensions[col].width = 22

    ws2 = wb.create_sheet("Interaction")
    ws2.append(["Scenario", "Pred-DWA @ A=off (pp)", "Pred-DWA @ A=on (pp)",
                "Heuristic @ B=off (pp)", "Heuristic @ B=on (pp)",
                "Interaction (pp)", "Interaction p", "Paired seeds"])
    for c in ws2[1]:
        c.font = bold
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for name, label, ft, _rows in results:
        ws2.append([label, round(ft["b_at_a0"], 1), round(ft["b_at_a1"], 1),
                    round(ft["a_at_b0"], 1), round(ft["a_at_b1"], 1),
                    round(ft["interaction"], 1), round(ft["interaction_p"], 4),
                    ft["n_seeds"]])
    for col in "ABCDEFGH":
        ws2.column_dimensions[col].width = 18

    wb.save(path)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "experiment_data_factorial.xlsx"))
    args = ap.parse_args()

    print("2x2 factorial ablation: %d runs x %d configs x (nominal + 3 structured) | %d workers"
          % (args.runs, len(CONFIGS), args.jobs))

    results = []

    print("\n########## nominal ##########")
    rows, dt = run_nominal(args.runs, args.jobs)
    ft = factorial_table(rows)
    print_table("nominal", "Nominal (12 static, 10 dynamic)", ft)
    results.append(("nominal", "Nominal", ft, rows))
    print("  elapsed %.0f s" % dt)

    for name, spec in SS.structured_specs().items():
        print("\n########## %s ##########" % name)
        rows, dt = run_structured(name, spec, args.runs, args.jobs)
        ft = factorial_table(rows)
        print_table(name, spec["name"], ft)
        results.append((name, spec["name"], ft, rows))
        print("  elapsed %.0f s" % dt)

    write_xlsx(results, args.out)
    print("\nWrote %s" % args.out)
    print("Hardware: 8 cores | Python %s" % sys.version.split()[0])


if __name__ == "__main__":
    main()
