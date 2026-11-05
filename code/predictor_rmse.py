"""
predictor_rmse.py -- position-RMSE comparison of the constant-velocity Kalman
filter vs. the adaptive coordinated-turn predictor (imm.py) on manoeuvring
obstacle trajectories, under the idealised noiseless measurement model of
Section 3.3.

The motion model mirrors Section 5.4: each obstacle turns at a rate sampled
uniformly in [-0.6, 0.6] rad/s and accelerates at a rate sampled uniformly in
[-0.4, 0.4] m/s^2, with speed clamped to [0.3, 3.0] m/s.  Reported in
Section 5.4 as a supporting measurement of the adaptive predictor's benefit.
"""
import math
import os

import numpy as np

import benchmark as B
import imm

DATA_DIR = os.path.join(os.path.dirname(B.HERE), "data")

DT = 0.2
N_STEPS = 250
HORIZONS = (1, 5, 15, 25)
N_TRAJ = 100


def _trajectory(seed):
    rng = np.random.default_rng(seed)
    x = y = 0.0
    ang = rng.uniform(0, 2 * math.pi)
    speed = rng.uniform(0.8, 1.4)
    turn = rng.uniform(-0.6, 0.6)
    accel = rng.uniform(-0.4, 0.4)
    truth = []
    for _ in range(N_STEPS):
        speed = max(0.3, min(3.0, speed + accel * DT))
        ang += turn * DT
        vx, vy = speed * math.cos(ang), speed * math.sin(ang)
        x += vx * DT
        y += vy * DT
        truth.append((x, y))
    return truth


def _rmse_one(seed):
    truth = _trajectory(seed)
    cv = B.A.KalmanFilter(DT); cv.initialize(0, 0, 0, 0)
    act = imm.AdaptiveCTFilter(DT, process_noise=B.KF_PROCESS_NOISE,
                               measurement_noise=B.KF_OBS_NOISE,
                               turn_rate_smoothing=1.0, max_turn_rate=1.2)
    act.initialize(0, 0, 0, 0)
    sq = {h: {"cv": 0.0, "act": 0.0} for h in HORIZONS}
    cnt = {h: 0 for h in HORIZONS}
    for t in range(N_STEPS):
        z = np.array(truth[t])
        for h in HORIZONS:
            if t + h < N_STEPS:
                cnt[h] += 1
        for f, key in ((cv, "cv"), (act, "act")):
            f.predict(DT)
            f.update(z)
            for h in HORIZONS:
                if t + h < N_STEPS:
                    ps, _ = f.predict_n_steps(h)
                    tx, ty = truth[t + h]
                    sq[h][key] += (ps[0] - tx) ** 2 + (ps[1] - ty) ** 2
    return {h: {k: sq[h][k] / cnt[h] for k in ("cv", "act")} for h in HORIZONS}


def main():
    agg = {h: {"cv": 0.0, "act": 0.0} for h in HORIZONS}
    for s in range(N_TRAJ):
        mse = _rmse_one(s)
        for h in HORIZONS:
            for k in ("cv", "act"):
                agg[h][k] += mse[h][k]
    print("%-8s %10s %10s %8s" % ("horizon", "CV-KF RMSE", "ACT-KF RMSE", "reduc%"))
    rmse = {}
    for h in HORIZONS:
        cv = math.sqrt(agg[h]["cv"] / N_TRAJ)
        act = math.sqrt(agg[h]["act"] / N_TRAJ)
        rmse[h] = (cv, act)
        print("%-8d %10.2f %10.2f %7.0f%%" % (h, cv, act, 100 * (1 - act / cv)))

    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Position RMSE"
    ws.append(["Manoeuvring trajectory", "horizon (steps)", "CV-KF RMSE (m)",
               "ACT-KF RMSE (m)", "reduction (%)"])
    for h in HORIZONS:
        cv, act = rmse[h]
        ws.append(["Section 5.4", h, round(cv, 3), round(act, 3),
                   round(100 * (1 - act / cv), 1)])
    out = os.path.join(DATA_DIR, "experiment_data_maneuver_rmse.xlsx")
    wb.save(out)
    print("\nWrote %s" % out)


if __name__ == "__main__":
    main()
