"""
factorial_maneuver.py -- 2x2 factorial on the manoeuvring and crossing scenes.

Review 2, round 2, comment #3: the factorial was run on the nominal and
structured scenes but not on the manoeuvring or crossing scenarios.  This
script runs the same four cells (adaptive heuristic A x prediction-aware DWA B)
on:

  * manoeuvring obstacles (turning/accelerating; adaptive coordinated-turn
    predictor, matching Section 5.4), and
  * crossing / head-on traffic (constant velocity, matching Section 5.5).

It reuses factorial_ablation.factorial_table / print_table / write_xlsx so the
statistics are identical to Table 5.

Run:
    py factorial_maneuver.py --runs 100 --jobs 7
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import benchmark as B
import scenario_study as SS
import factorial_ablation as FA

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

CONFIGS = FA.CONFIGS  # astar_dwa, astar_pred_dwa, proposed, proposed_pred_dwa


def _init_maneuver():
    B.PREDICTOR = "act"


def _init_crossing(spec):
    B.START = (10.0, 10.0)
    B.GOAL = (180.0, 180.0)
    B.BOUNDS = (0, 200, 0, 200)
    B.GRID_SIZE = (100, 100)
    B.RESOLUTION = 2.0
    B.STATIC_OBSTACLES = spec["static"]
    B.build_scenario = lambda seed, maneuver=False: SS.build_spec_scenario(spec, seed)
    B.PREDICTOR = "cv"


def _worker(job):
    cfg, seed, maneuver = job
    try:
        return B.run_episode(cfg, seed, B.N_PRED_STEPS, maneuver=maneuver)
    except Exception as exc:
        return dict(config=cfg, seed=seed, success=0, outcome="error:%s" % exc,
                    path_length=float("nan"), mean_risk=float("nan"),
                    max_risk=float("nan"), tracking_error=float("nan"),
                    replan_time=float("nan"), n_replans=0, sim_time=0.0,
                    full_step_time=float("nan"))


def _run(jobs_list, jobs, init_fn=None, init_args=None):
    if init_fn is None:
        with mp.Pool(jobs) as pool:
            return [r for r in pool.imap_unordered(_worker, jobs_list)]
    with mp.Pool(jobs, initializer=init_fn, initargs=init_args or ()) as pool:
        return [r for r in pool.imap_unordered(_worker, jobs_list)]


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default=os.path.join(
        DATA_DIR, "experiment_data_factorial_maneuver_crossing.xlsx"))
    args = ap.parse_args()

    print("2x2 factorial on manoeuvring + crossing: %d runs x %d configs | %d workers"
          % (args.runs, len(CONFIGS), args.jobs))

    results = []
    t0 = time.time()

    print("\n########## manoeuvring ##########")
    jobs_list = [(c, s, True) for c in CONFIGS for s in range(args.runs)]
    rows = _run(jobs_list, args.jobs, _init_maneuver)
    ft = FA.factorial_table(rows)
    FA.print_table("maneuver", "Manoeuvring obstacles (adaptive CT predictor)", ft)
    results.append(("maneuver", "Manoeuvring obstacles", ft, rows))
    print("  elapsed %.0f s" % (time.time() - t0))

    print("\n########## crossing / head-on ##########")
    spec = SS.crossing_specs()["crossing"]
    jobs_list = [(c, s, False) for c in CONFIGS for s in range(args.runs)]
    rows = _run(jobs_list, args.jobs, _init_crossing, (spec,))
    ft = FA.factorial_table(rows)
    FA.print_table("crossing", "Crossing / head-on traffic", ft)
    results.append(("crossing", "Crossing / head-on traffic", ft, rows))
    print("  elapsed %.0f s" % (time.time() - t0))

    FA.write_xlsx(results, args.out)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
