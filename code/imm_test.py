"""
imm_test.py -- sanity check that the adaptive coordinated-turn predictor
reduces to CV on straight-line motion and beats the plain CV-Kalman filter on
turning/accelerating motion.
"""
import math

import numpy as np

import benchmark as B
import imm

DT = 0.2
N_STEPS = 250  # 50 s
HORIZONS = [1, 5, 10, 25]


def cv_kf():
    kf = B.A.KalmanFilter(DT)
    kf.initialize(0.0, 0.0, 0.0, 0.0)
    return kf


def act_kf():
    kf = imm.AdaptiveCTFilter(DT, process_noise=0.5, measurement_noise=2.0,
                              turn_rate_smoothing=1.0, max_turn_rate=1.2)
    kf.initialize(0.0, 0.0, 0.0, 0.0)
    return kf


def run_forward(name, kind, seed=0):
    rng = np.random.default_rng(seed)
    x, y = 0.0, 0.0
    heading = rng.uniform(0, 2 * math.pi)
    speed = 1.5
    vx, vy = speed * math.cos(heading), speed * math.sin(heading)
    turn_rate, accel = 0.0, 0.0
    if kind == "maneuver":
        turn_rate, accel = 0.5, 0.3
    truth = []
    for _ in range(N_STEPS):
        heading += turn_rate * DT
        speed = max(0.3, min(3.0, speed + accel * DT))
        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
        x += vx * DT
        y += vy * DT
        truth.append((x, y))

    f_cv = cv_kf()
    f_act = act_kf()
    errors = {h: {"cv": [], "act": []} for h in HORIZONS}

    for t in range(N_STEPS):
        z = np.array(truth[t])
        for f, key in ((f_cv, "cv"), (f_act, "act")):
            f.predict(DT)
            f.update(z)
            for h in HORIZONS:
                if t + h < N_STEPS:
                    ps, _ = f.predict_n_steps(h)
                    tx, ty = truth[t + h]
                    errors[h][key].append(math.hypot(ps[0] - tx, ps[1] - ty))

    print("\n== %s (%s) ==" % (name, kind))
    print("%-8s %12s %12s" % ("horizon", "CV-KF RMSE", "ACT-KF RMSE"))
    for h in HORIZONS:
        cv_r = float(np.sqrt(np.mean(np.array(errors[h]["cv"]) ** 2)))
        im_r = float(np.sqrt(np.mean(np.array(errors[h]["act"]) ** 2)))
        print("%-8d %12.3f %12.3f" % (h, cv_r, im_r))
    print("final estimated turn rate = %.4f (truth %s)"
          % (f_act.omega, turn_rate))
    return errors


if __name__ == "__main__":
    run_forward("straight-line", "cv")
    run_forward("maneuvering", "maneuver")
