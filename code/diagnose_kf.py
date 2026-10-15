"""diagnose_kf.py -- check whether the KF obstacle prediction is accurate.

The paper's risk field and the prediction-aware DWA both rely on
`KalmanFilter.predict_n_steps`. This script simulates the tracker for a few
steps and compares the k-step-ahead prediction against the *actual* future
obstacle position (available because obstacles move deterministically at
constant velocity, bouncing off the map bounds).
"""

import math
import numpy as np

import benchmark as B


def simulate_obstacle_forward(obs, k, dt, bounds):
    x, y = obs.x, obs.y
    vx, vy = obs.vx, obs.vy
    for _ in range(k):
        x += vx * dt
        y += vy * dt
        if x < bounds[0] + obs.radius:
            x = bounds[0] + obs.radius
            vx = abs(vx)
        if x > bounds[1] - obs.radius:
            x = bounds[1] - obs.radius
            vx = -abs(vx)
        if y < bounds[2] + obs.radius:
            y = bounds[2] + obs.radius
            vy = abs(vy)
        if y > bounds[3] - obs.radius:
            y = bounds[3] - obs.radius
            vy = -abs(vy)
    return x, y


def main():
    env, uav = B.build_scenario(0)
    dt = B.DT
    horizon = 7  # 1.4 s, the DWA prediction window

    errs = []
    for t in range(200):  # 40 s of simulation
        env.update_dynamic(dt)
        env.update_kf_trackers(t * dt)
        for obs, tracker in zip(env.dynamic_obstacles, env.obstacle_trackers):
            ps, _ = tracker.get_predicted_state(horizon)
            ax, ay = simulate_obstacle_forward(obs, horizon, dt, B.BOUNDS)
            errs.append(math.hypot(ps[0] - ax, ps[1] - ay))

    errs = np.array(errs)
    print("Prediction error over horizon=%d steps (%.1f s):" % (horizon, horizon * dt))
    print("  mean  = %.3f m" % errs.mean())
    print("  median= %.3f m" % np.median(errs))
    print("  p90   = %.3f m" % np.percentile(errs, 90))
    print("  max   = %.3f m" % errs.max())
    print("\nReference: obstacle radius 3-5 m, UAV radius 1 m, collision margin 0.5 m.")
    print("A prediction error larger than ~(r_obs + 1.5) m is enough to mislead DWA.")


if __name__ == "__main__":
    main()
