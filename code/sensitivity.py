"""sensitivity.py -- parameter-sensitivity sweeps for the proposed algorithm.

Sweeps the prediction horizon, the adaptive-heuristic steepness beta, and the
Kalman-filter process/observation noise variances, reporting the success rate
(and mean risk) of the proposed configuration at each value. This addresses
Review 2 #32 (parameter sensitivity), #33 (prediction horizon) and #34 (noise
robustness).

Run:  py sensitivity.py
"""

import multiprocessing as mp

import numpy as np

import benchmark as B

RUNS = 30
CONFIG = "proposed"
JOBS = 6
NOM_BETA, NOM_PROC, NOM_OBS = 10.0, 0.5, 2.0


def _init(beta, proc, obs):
    # Runs in each worker process; sets the benchmark module globals that
    # run_episode and the patched KalmanFilter.__init__ read.
    B.BETA = beta
    B.KF_PROCESS_NOISE = proc
    B.KF_OBS_NOISE = obs


def _worker(seed_pred):
    seed, pred_steps = seed_pred
    return B.run_episode(CONFIG, seed, pred_steps, maneuver=False)


def run_batch(pred_steps, beta, proc, obs, label):
    jobs = [(s, pred_steps) for s in range(RUNS)]
    with mp.Pool(JOBS, initializer=_init, initargs=(beta, proc, obs)) as pool:
        rows = pool.map(_worker, jobs)
    succ = sum(r["success"] for r in rows)
    rate = 100.0 * succ / RUNS
    risks = [r["mean_risk"] for r in rows
             if r["success"] and not np.isnan(r["mean_risk"])]
    risk = float(np.mean(risks)) if risks else float("nan")
    print("%-32s success=%5.1f%%  meanRisk=%.4f  (collisions=%d)"
          % (label, rate, risk, sum(1 for r in rows if r["outcome"] == "collision")))
    return rate


if __name__ == "__main__":
    print("Sweeping '%s' configuration, %d seeds per point, %d workers\n"
          % (CONFIG, RUNS, JOBS))

    print("--- prediction horizon (steps) ---")
    for ps in [5, 10, 15, 25, 40]:
        run_batch(ps, NOM_BETA, NOM_PROC, NOM_OBS,
                  "horizon=%d steps (%.1f s)" % (ps, ps * B.DT))

    print("\n--- adaptive-heuristic steepness beta ---")
    for beta in [1.0, 5.0, 10.0, 20.0, 50.0]:
        run_batch(B.N_PRED_STEPS, beta, NOM_PROC, NOM_OBS, "beta=%.0f" % beta)

    print("\n--- Kalman observation noise variance R ---")
    for obs in [0.5, 2.0, 5.0, 10.0]:
        run_batch(B.N_PRED_STEPS, NOM_BETA, NOM_PROC, obs, "obs_noise=%.1f" % obs)

    print("\n--- Kalman process noise variance Q ---")
    for proc in [0.1, 0.5, 2.0, 5.0]:
        run_batch(B.N_PRED_STEPS, NOM_BETA, proc, NOM_OBS, "proc_noise=%.1f" % proc)
