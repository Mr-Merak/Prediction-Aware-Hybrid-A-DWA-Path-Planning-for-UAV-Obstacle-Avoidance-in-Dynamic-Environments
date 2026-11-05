"""
imm.py -- Adaptive coordinated-turn obstacle-motion predictor.

Drop-in replacement for `KalmanFilter` (11.py) sharing the same interface:
    dt, initialized
    initialize(x, y, vx, vy)
    predict(dt=None)
    update(z)
    predict_n_steps(n)
    get_predicted_pos_cov(n_steps)

The original predictor assumes constant velocity, so its multi-step prediction
is a straight-line extrapolation.  When an obstacle turns or accelerates this
assumption breaks and the propagated risk field / prediction-aware DWA forward
simulation are wrong.  This module augments the constant-velocity Kalman filter
with an *online turn-rate estimate*: the turn rate is recovered from the change
in the estimated velocity direction between successive steps and smoothed with
a first-order low-pass filter.  The n-step prediction then propagates a
coordinated-turn motion model parameterised by the estimated turn rate, instead
of the fixed straight-line transition.

Key property: when the estimated turn rate is ~0, the coordinated-turn
transition reduces exactly to the constant-velocity transition, so on
non-manoeuvring obstacles the predictions are unchanged and the nominal
results are unaffected; only strongly turning/accelerating obstacles trigger a
materially different (curved) prediction.  Residual speed changes are absorbed
by the discrete-white-noise-acceleration process noise, exactly as before.
"""

import math

import numpy as np

_TINY = 1e-9


class AdaptiveCTFilter:
    def __init__(self, dt, process_noise=0.5, measurement_noise=2.0,
                 turn_rate_smoothing=1.0, max_turn_rate=1.2):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.turn_rate_smoothing = turn_rate_smoothing  # seconds
        self.max_turn_rate = max_turn_rate

        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
        self.R = np.eye(2) * measurement_noise

        self.x = np.zeros(4)
        self.P = np.eye(4) * 5.0
        self.F = self._F_cv(dt)
        self.Q = self._Q(dt)

        self.omega = 0.0            # estimated turn rate (rad/s)
        self.alpha = dt / (dt + turn_rate_smoothing)
        self._prev_v = None

        self.initialized = False

    # ------------------------------------------------------------------
    @staticmethod
    def _F_cv(dt):
        return np.array([[1.0, 0.0, dt, 0.0],
                         [0.0, 1.0, 0.0, dt],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

    def _Q(self, dt):
        q = self.process_noise
        Q = np.eye(4) * q
        Q[0, 0] = q * dt ** 4 / 4.0
        Q[1, 1] = q * dt ** 4 / 4.0
        Q[2, 2] = q * dt ** 2
        Q[3, 3] = q * dt ** 2
        return Q

    def _F_ct(self, w, n=1):
        """Coordinated-turn transition for turn rate w over n*dt seconds."""
        wdt = w * (n * self.dt)
        if abs(wdt) < _TINY:
            return self._F_cv(n * self.dt)
        sw, cw = math.sin(wdt), math.cos(wdt)
        return np.array([
            [1.0, 0.0, sw / w, (cw - 1.0) / w],
            [0.0, 1.0, (1.0 - cw) / w, sw / w],
            [0.0, 0.0, cw, -sw],
            [0.0, 0.0, sw, cw]])

    # ------------------------------------------------------------------
    def initialize(self, x, y, vx=0.0, vy=0.0):
        self.x = np.array([x, y, vx, vy], dtype=float)
        self.P = np.eye(4) * 5.0
        self.omega = 0.0
        self._prev_v = None
        self.initialized = True

    # ------------------------------------------------------------------
    def predict(self, dt=None):
        if dt is not None:
            self.dt = dt
            self.F = self._F_cv(dt)
            self.Q = self._Q(dt)
            self.alpha = dt / (dt + self.turn_rate_smoothing)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        if not self.initialized:
            self.initialize(z[0], z[1])
            return
        z = np.asarray(z, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # Online turn-rate estimate from the change in velocity direction.
        v = self.x[2:4]
        if self._prev_v is not None:
            cross = self._prev_v[0] * v[1] - self._prev_v[1] * v[0]
            dot = self._prev_v[0] * v[0] + self._prev_v[1] * v[1]
            denom = self.dt * math.hypot(self._prev_v[0], self._prev_v[1]) \
                    * math.hypot(v[0], v[1])
            if denom > _TINY:
                omega_inst = math.atan2(cross, dot) / self.dt
                omega_inst = max(-self.max_turn_rate,
                                 min(self.max_turn_rate, omega_inst))
                self.omega += self.alpha * (omega_inst - self.omega)
        self._prev_v = v.copy()

    # ------------------------------------------------------------------
    def predict_n_steps(self, n):
        """n-step prediction using the coordinated-turn model with the
        estimated turn rate (reduces to constant-velocity when omega ~ 0)."""
        Fn = self._F_ct(self.omega, n)
        x_pred = Fn @ self.x
        P_pred = Fn @ self.P @ Fn.T + self.Q * n
        return x_pred, P_pred

    def get_predicted_pos_cov(self, n_steps):
        _, P_pred = self.predict_n_steps(n_steps)
        return P_pred[0:2, 0:2]
