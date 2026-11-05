"""
learned_predictor.py -- Reviewer 2 suggestion #3: a learned trajectory-prediction
baseline (numpy GRU) compared against a constant-velocity Kalman filter and naive
constant-velocity extrapolation.

The obstacle motion models mirror the manuscript: (i) constant-velocity with
boundary bouncing (Section 5.2, nominal), and (ii) turning/accelerating
manoeuvring motion (Section 5.4). Position measurements are corrupted with
additive Gaussian noise (R = 2.0, i.e., ~1.41 m per axis), matching the paper's
observation model. At each time step each predictor observes the noisy positions
up to t and predicts the true positions t+1 .. t+H.

Predictors:
  CV-extrapolation : velocity from the last two noisy observations, straight-line.
  Kalman (CV)      : the paper's constant-velocity KF (Q=0.5, R=2.0).
  GRU (learned)    : a small GRU trained on noisy->true next-position windows,
                     rolled out autoregressively for multi-step prediction.

Metric: RMSE of predicted position vs ground truth, as a function of the
prediction horizon (kappa = 1, 5, 15, 25 steps => 0.2 s .. 5.0 s).

Run:
    py learned_predictor.py --trains 300 --tests 100 --epochs 300
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

DT = 0.2
BOUNDS = (0.0, 200.0)
R_OBS = 2.0          # observation-noise variance (paper Section 5.1 / 5.10, R = 2.0)
Q_PROC = 0.5         # process-noise variance (paper Section 5.1)
NOISE_STD = math.sqrt(R_OBS)


# --------------------------------------------------------------------------
# Motion models
# --------------------------------------------------------------------------
def gen_cv_trajectory(steps, seed):
    rng = np.random.default_rng(seed)
    x = float(rng.uniform(BOUNDS[0] + 20, BOUNDS[1] - 20))
    y = float(rng.uniform(BOUNDS[0] + 20, BOUNDS[1] - 20))
    ang = float(rng.uniform(0, 2 * math.pi))
    speed = float(rng.uniform(0.8, 1.4))
    vx, vy = speed * math.cos(ang), speed * math.sin(ang)
    true, obs = [], []
    for _ in range(steps):
        x += vx * DT
        y += vy * DT
        if x < BOUNDS[0] + 5:
            x = BOUNDS[0] + 5; vx = abs(vx)
        elif x > BOUNDS[1] - 5:
            x = BOUNDS[1] - 5; vx = -abs(vx)
        if y < BOUNDS[0] + 5:
            y = BOUNDS[0] + 5; vy = abs(vy)
        elif y > BOUNDS[1] - 5:
            y = BOUNDS[1] - 5; vy = -abs(vy)
        true.append((x, y))
        obs.append((x + rng.normal(0, NOISE_STD), y + rng.normal(0, NOISE_STD)))
    return np.array(true), np.array(obs)


def gen_maneuver_trajectory(steps, seed):
    rng = np.random.default_rng(seed)
    x = float(rng.uniform(BOUNDS[0] + 20, BOUNDS[1] - 20))
    y = float(rng.uniform(BOUNDS[0] + 20, BOUNDS[1] - 20))
    ang = float(rng.uniform(0, 2 * math.pi))
    speed = float(rng.uniform(0.8, 1.4))
    turn = float(rng.uniform(-0.6, 0.6))
    accel = float(rng.uniform(-0.4, 0.4))
    vx, vy = speed * math.cos(ang), speed * math.sin(ang)
    true, obs = [], []
    for _ in range(steps):
        speed = max(0.3, min(3.0, speed + accel * DT))
        ang += turn * DT
        vx, vy = speed * math.cos(ang), speed * math.sin(ang)
        x += vx * DT
        y += vy * DT
        if x < BOUNDS[0] + 5:
            x = BOUNDS[0] + 5; vx = abs(vx)
        elif x > BOUNDS[1] - 5:
            x = BOUNDS[1] - 5; vx = -abs(vx)
        if y < BOUNDS[0] + 5:
            y = BOUNDS[0] + 5; vy = abs(vy)
        elif y > BOUNDS[1] - 5:
            y = BOUNDS[1] - 5; vy = -abs(vy)
        true.append((x, y))
        obs.append((x + rng.normal(0, NOISE_STD), y + rng.normal(0, NOISE_STD)))
    return np.array(true), np.array(obs)


# --------------------------------------------------------------------------
# Predictors
# --------------------------------------------------------------------------
class CVKalman:
    """Constant-velocity Kalman filter matching fixes.py (Q=0.5, R=2.0)."""

    def __init__(self, dt=DT):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * Q_PROC
        self.Q[0, 0] = Q_PROC * dt ** 4 / 4
        self.Q[1, 1] = Q_PROC * dt ** 4 / 4
        self.Q[2, 2] = Q_PROC * dt ** 2
        self.Q[3, 3] = Q_PROC * dt ** 2
        self.R = np.eye(2) * R_OBS
        self.x = np.zeros(4)
        self.P = np.eye(4) * 5.0

    def update(self, z):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = np.asarray(z) - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def predict(self, k):
        return (np.linalg.matrix_power(self.F, k) @ self.x)[:2]


# --------------------------------------------------------------------------
# Numpy GRU
# --------------------------------------------------------------------------
def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))


class GRUPredictor:
    """A minimal GRU mapping a window of noisy positions to the next true position."""

    def __init__(self, in_dim=2, hidden=16, out_dim=2, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / math.sqrt(hidden)
        self.Wz = rng.normal(0, s, (hidden, in_dim))
        self.Wr = rng.normal(0, s, (hidden, in_dim))
        self.Wh = rng.normal(0, s, (hidden, in_dim))
        self.Uz = rng.normal(0, s, (hidden, hidden))
        self.Ur = rng.normal(0, s, (hidden, hidden))
        self.Uh = rng.normal(0, s, (hidden, hidden))
        self.bz = np.zeros(hidden)
        self.br = np.zeros(hidden)
        self.bh = np.zeros(hidden)
        self.Wo = rng.normal(0, s, (out_dim, hidden))
        self.bo = np.zeros(out_dim)

    def forward(self, xs):
        """xs: (T, in). Returns (y, cache). y is predicted next position (out_dim,)."""
        T = xs.shape[0]
        h = np.zeros(self.Wz.shape[0])
        cache = dict(xs=xs, zs=[], rs=[], hs=[h.copy()], hts=[], azs=[], ars=[], ahs=[])
        for t in range(T):
            x = xs[t]
            az = self.Wz @ x + self.Uz @ h + self.bz
            z = _sigmoid(az)
            ar = self.Wr @ x + self.Ur @ h + self.br
            r = _sigmoid(ar)
            ah = self.Wh @ x + self.Uh @ (r * h) + self.bh
            ht = np.tanh(ah)
            h = (1.0 - z) * h + z * ht
            cache["azs"].append(az); cache["ars"].append(ar); cache["ahs"].append(ah)
            cache["zs"].append(z); cache["rs"].append(r)
            cache["hts"].append(ht); cache["hs"].append(h.copy())
        y = self.Wo @ h + self.bo
        cache["y"] = y
        return y, cache

    def _backward(self, cache, dy):
        """Return gradient dict given dy = dL/dy (out_dim,)."""
        T = cache["xs"].shape[0]
        g = {k: np.zeros_like(getattr(self, k)) for k in
             ["Wz", "Wr", "Wh", "Uz", "Ur", "Uh", "bz", "br", "bh", "Wo", "bo"]}
        h_T = cache["hs"][-1]
        g["Wo"] += np.outer(dy, h_T)
        g["bo"] += dy
        dh = self.Wo.T @ dy
        for t in reversed(range(T)):
            x = cache["xs"][t]
            h_prev = cache["hs"][t]      # h_{t-1}
            z = cache["zs"][t]; r = cache["rs"][t]; ht = cache["hts"][t]
            az = cache["azs"][t]; ar = cache["ars"][t]; ah = cache["ahs"][t]
            dht = dh * z
            dz = dh * (ht - h_prev)
            dh_prev = dh * (1.0 - z)
            dah = dht * (1.0 - np.tanh(ah) ** 2)
            dr = self.Uh.T @ dah * h_prev
            dar = dr * (r * (1.0 - r))
            daz = dz * (z * (1.0 - z))
            dh_prev = dh_prev + self.Uz.T @ daz + self.Ur.T @ dar + self.Uh.T @ (dah * r)
            g["Wh"] += np.outer(dah, x); g["Uh"] += np.outer(dah, r * h_prev); g["bh"] += dah
            g["Wr"] += np.outer(dar, x); g["Ur"] += np.outer(dar, h_prev); g["br"] += dar
            g["Wz"] += np.outer(daz, x); g["Uz"] += np.outer(daz, h_prev); g["bz"] += daz
            dh = dh_prev
        return g

    def train_step(self, xs_batch, ytrue_batch, lr, adam):
        # xs_batch: (B, T, in), ytrue_batch: (B, out)
        b = xs_batch.shape[0]
        total_loss = 0.0
        grads = {k: np.zeros_like(getattr(self, k)) for k in adam}
        for i in range(b):
            y, cache = self.forward(xs_batch[i])
            dy = y - ytrue_batch[i]
            total_loss += 0.5 * float(np.sum(dy ** 2))
            g = self._backward(cache, dy)
            for k in grads:
                grads[k] += g[k]
        for k in grads:
            g = grads[k] / b
            m, v, t = adam[k]
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g ** 2
            mhat = m / (1 - 0.9 ** t)
            vhat = v / (1 - 0.999 ** t)
            setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + 1e-8))
            adam[k] = (m, v, t + 1)
        return total_loss / b

    def predict_one(self, window):
        y, _ = self.forward(window)
        return y


def train_gru(X_train, Y_train, hidden=16, epochs=300, lr=0.01, batch=64, seed=0, verbose=False):
    model = GRUPredictor(in_dim=X_train.shape[2], hidden=hidden, out_dim=Y_train.shape[1], seed=seed)
    adam = {k: (np.zeros_like(getattr(model, k)), np.zeros_like(getattr(model, k)), 1)
            for k in ["Wz", "Wr", "Wh", "Uz", "Ur", "Uh", "bz", "br", "bh", "Wo", "bo"]}
    n = X_train.shape[0]
    rng = np.random.default_rng(1)
    idx = np.arange(n)
    for ep in range(epochs):
        rng.shuffle(idx)
        losses = []
        for b in range(0, n, batch):
            bidx = idx[b:b + batch]
            loss = model.train_step(X_train[bidx], Y_train[bidx], lr, adam)
            losses.append(loss)
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print("    epoch %3d  loss %.5f" % (ep, float(np.mean(losses))), flush=True)
    return model


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------
def build_dataset(gen_fn, n_traj, steps, window, normalize=100.0):
    """Return (X, Y) of shape (samples, window, 2) and (samples, 2)."""
    X, Y = [], []
    for s in range(n_traj):
        true, obs = gen_fn(steps, 1000 + s)
        z = obs / normalize - 1.0      # input: normalized noisy positions
        p = true / normalize - 1.0     # target: normalized true positions
        for t in range(window, steps):
            X.append(z[t - window:t])
            Y.append(p[t])             # next true position after window
    return np.array(X), np.array(Y)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
HORIZONS = [1, 5, 15, 25]


def evaluate_predictors(trajectories, gru, window, normalize=100.0):
    """Return dict predictor -> {kappa: RMSE} over all test trajectories."""
    n_tr = len(trajectories)
    rmse = {k: {p: 0.0 for p in ["cv", "kf", "gru"]} for k in HORIZONS}
    counts = {k: 0 for k in HORIZONS}
    for true, obs in trajectories:
        kf = CVKalman()
        kf_hist = []
        for t in range(len(obs)):
            kf.update(obs[t])
            kf_hist.append(kf.x.copy())
            if t >= window:
                window_z = obs[t - window + 1:t + 1]
                for k in HORIZONS:
                    if t + k >= len(obs):
                        continue
                    ptrue = true[t + k]
                    # naive CV extrapolation
                    v_est = (obs[t] - obs[t - 1]) / DT
                    cv_pred = obs[t] + k * v_est * DT
                    # Kalman
                    kf_pred = kf.predict(k)
                    # GRU (autoregressive rollout)
                    w = (window_z / normalize - 1.0).astype(float)
                    if k == 1:
                        gpred_norm = gru.predict_one(w)
                    else:
                        wcur = w.copy()
                        gpred_norm = None
                        for _ in range(k):
                            gpred_norm = gru.predict_one(wcur)
                            wcur = np.vstack([wcur[1:], gpred_norm.reshape(1, -1)])
                    gpred = (gpred_norm + 1.0) * normalize
                    rmse[k]["cv"] += np.sum((cv_pred - ptrue) ** 2)
                    rmse[k]["kf"] += np.sum((kf_pred - ptrue) ** 2)
                    rmse[k]["gru"] += np.sum((gpred - ptrue) ** 2)
                    counts[k] += 1
    out = {}
    for k in HORIZONS:
        out[k] = {p: math.sqrt(rmse[k][p] / counts[k]) for p in ["cv", "kf", "gru"]}
    return out


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--trains", type=int, default=300)
    ap.add_argument("--tests", type=int, default=100)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "experiment_data_learned_predictor.xlsx"))
    args = ap.parse_args()

    print("Building datasets (CV + manoeuvring) ...")
    t0 = time.time()
    Xc, Yc = build_dataset(gen_cv_trajectory, args.trains, args.steps, args.window)
    Xm, Ym = build_dataset(gen_maneuver_trajectory, args.trains, args.steps, args.window)
    print("  samples: CV %d, manoeuvre %d  (%.0f s)" % (len(Xc), len(Xm), time.time() - t0))

    print("\nTraining GRU on constant-velocity trajectories ...")
    gru_cv = train_gru(Xc, Yc, hidden=args.hidden, epochs=args.epochs, verbose=True)
    print("Training GRU on manoeuvring trajectories ...")
    gru_man = train_gru(Xm, Ym, hidden=args.hidden, epochs=args.epochs, verbose=True)

    test_cv = [gen_cv_trajectory(args.steps, 5000 + s) for s in range(args.tests)]
    test_man = [gen_maneuver_trajectory(args.steps, 5000 + s) for s in range(args.tests)]

    print("\nEvaluating on CV test set ...")
    res_cv = evaluate_predictors(test_cv, gru_cv, args.window)
    print("Evaluating on manoeuvring test set ...")
    res_man = evaluate_predictors(test_man, gru_man, args.window)

    print("\n===== Prediction RMSE (m) =====")
    print("%-16s %6s %8s %8s %8s" % ("Predictor", "k=1", "k=5", "k=15", "k=25"))
    for name, res in [("CV (CV scene)", res_cv), ("Maneuver scene", res_man)]:
        print("-- %s --" % name)
        for p, label in [("cv", "CV extrap"), ("kf", "Kalman (CV)"), ("gru", "GRU (learned)")]:
            print("  %-12s %6.2f %8.2f %8.2f %8.2f"
                  % (label, res[1][p], res[5][p], res[15][p], res[25][p]))

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, res, title in [(axes[0], res_cv, "Constant-velocity motion"),
                           (axes[1], res_man, "Manoeuvring motion")]:
        h = HORIZONS
        ax.plot(h, [res[k]["cv"] for k in h], "s--", color="C1", label="CV extrapolation")
        ax.plot(h, [res[k]["kf"] for k in h], "o-", color="C0", label="Kalman filter (CV)")
        ax.plot(h, [res[k]["gru"] for k in h], "^-", color="C2", label="GRU (learned)")
        ax.set_xlabel(r"Prediction horizon $\kappa$ (steps of 0.2 s)")
        ax.set_ylabel("Position RMSE (m)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Learned vs. model-based obstacle-trajectory prediction")
    fig_path = os.path.join(FIG_DIR, "figure_learned_predictor.png")
    fig.savefig(fig_path, dpi=130)
    print("\nWrote %s" % fig_path)

    # ---- xlsx ----
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    bold = Font(bold=True)
    ws = wb.active
    ws.title = "Prediction RMSE"
    ws.append(["Scenario", "Predictor", "RMSE k=1 (m)", "RMSE k=5 (m)",
               "RMSE k=15 (m)", "RMSE k=25 (m)"])
    for c in ws[1]:
        c.font = bold
    for scene, res in [("CV", res_cv), ("Manoeuvring", res_man)]:
        for p, label in [("cv", "CV extrapolation"), ("kf", "Kalman (CV)"), ("gru", "GRU (learned)")]:
            ws.append([scene, label, round(res[1][p], 3), round(res[5][p], 3),
                       round(res[15][p], 3), round(res[25][p], 3)])
    for col in "ABCDEF":
        ws.column_dimensions[col].width = 20
    wb.save(args.out)
    print("Wrote %s" % args.out)


if __name__ == "__main__":
    main()
