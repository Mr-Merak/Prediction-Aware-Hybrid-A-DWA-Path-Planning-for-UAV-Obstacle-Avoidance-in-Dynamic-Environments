"""
tuning.py -- Make the implementation match the manuscript's stated formulas.

These are specification-alignment changes, not performance tuning: the paper's
equations describe one thing and 11.py implemented another. Every change is
traceable to a specific equation in the manuscript.

--------------------------------------------------------------------------
T1 -- Risk-adaptive heuristic (manuscript Eq. 7-10)
--------------------------------------------------------------------------
Manuscript:
    f(n) = g(n) + (1 - alpha(rbar)) h_dist(n) + alpha(rbar) h_risk(n)   (7)
    alpha(rbar) = 1 / (1 + exp(-beta (rbar - r0)))                       (8)
    h_dist(n) = || p_n - p_goal ||_2                                     (9)
    h_risk(n) = rbar * d_max                                             (10)

11.py instead used
    alpha = 0.5 + 1.5*sigmoid,  beta_w = 2.0 - 2.0*sigmoid,
    h = alpha*basic_h + beta_w*risk_h,   risk_h = risk_val * basic_h

i.e. a non-convex double weight with `risk_h` scaled by the distance-to-goal
rather than the map diagonal, and a weight that can reach 2.0 (breaking
admissibility, contradicting the manuscript's guarantee in Section 3).

Fix: implement (7)-(10) verbatim. beta = 10 (steepness), r0 = 0.6 (the risk
threshold of Section 5.1), d_max = map diagonal.
"""

import math


def _d_max_of(env):
    b = env.bounds
    return math.hypot(b[1] - b[0], b[3] - b[2])


def adaptive_heuristic(self, node, goal, risk_val):
    h_dist = self.heuristic(node, goal)
    h_risk = risk_val * self.d_max
    # Eq. (8), numerically stable
    try:
        alpha = 1.0 / (1.0 + math.exp(-self.beta * (risk_val - self.r0)))
    except OverflowError:
        alpha = 1.0 if risk_val > self.r0 else 0.0
    return (1.0 - alpha) * h_dist + alpha * h_risk


def apply_all(algorithms_module):
    cls = algorithms_module.AdaptiveAStar
    cls.adaptive_heuristic = adaptive_heuristic

    # Inject the parameters the new heuristic needs (beta, r0, d_max).
    # `k` and `R0` already hold 10.0 and 0.6 respectively in 11.py; make the
    # intent explicit and add d_max.
    orig_init = cls.__init__

    def __init__(self, env, uav, goal, risk_threshold=0.6, k=10.0):
        orig_init(self, env, uav, goal, risk_threshold=risk_threshold, k=k)
        self.beta = k
        self.r0 = risk_threshold
        self.d_max = _d_max_of(env)

    cls.__init__ = __init__
    return algorithms_module
