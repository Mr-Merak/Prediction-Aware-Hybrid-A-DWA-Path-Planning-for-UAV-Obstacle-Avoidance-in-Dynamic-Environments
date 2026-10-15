# Prediction-Aware Hybrid A*--DWA Path Planning for UAV Obstacle Avoidance in Dynamic Environments

Simulation code and data for the manuscript:

**"Prediction-Aware Hybrid A*--DWA Path Planning for UAV Obstacle Avoidance in Dynamic Environments"**

Yanxin Zhang, Tian Jing, Meigen Huang, Tao Wang, Mengjun Ming  
College of Systems Engineering, National University of Defense Technology, Changsha 410073, China

## Overview

This repository provides the code and data supporting the paper. The work proposes a prediction--planning--avoidance framework that injects Kalman-filter motion prediction into both planning layers of a hybrid A*--DWA planner:

- **Prediction layer** -- a constant-velocity Kalman filter predicts obstacle trajectories and builds a spatiotemporal risk field whose covariance grows with the prediction horizon, propagating prediction uncertainty into a forward-looking cost map.
- **Planning layer** -- a risk-adaptive (Sigmoid-type) heuristic for the A* evaluation function, plus path smoothing via redundant-waypoint removal and cubic-spline interpolation.
- **Avoidance layer** -- a prediction-aware Dynamic Window Approach (DWA) whose forward simulation propagates the predicted obstacle motion instead of treating obstacles as static, with risk-triggered mode switching between the global and local layers.

All components are evaluated under a constrained-kinematics UAV model (linear/angular velocity and acceleration limits).

## Contents

- `code/` -- simulation code
  - `benchmark.py` -- main experiment framework (10 configurations, paired seeds, Wilson confidence intervals, exact McNemar tests, paired t/Wilcoxon tests, failure decomposition, acceleration-constrained UAV model).
  - `scenario_study.py` -- obstacle-density scalability, structured-environment (narrow passage / bottleneck / dead end), and crossing / head-on studies.
  - `sensitivity.py` -- parameter-sensitivity sweep.
  - `make_figures.py` -- regenerates Figures 3--8.
  - `dstar_lite.py` -- D* Lite baseline.
  - `diagnose_kf.py` -- Kalman-filter prediction-error diagnostic.
  - `11.py`, `fixes.py`, `tuning.py`, `patchtest.py` -- algorithm implementation and the applied bug fixes (FIX 1--7).
  - `路径规划.bib` -- reference library.
- `data/` -- experimental results (per-configuration summary and per-trial records)
  - `experiment_data_v4.xlsx` -- nominal scenario (100 paired seeds).
  - `experiment_data_maneuver.xlsx` -- maneuvering-obstacle scenario.
  - `experiment_data_density.xlsx` -- obstacle-density scalability.
  - `experiment_data_structured.xlsx` -- structured environments (narrow passage / bottleneck / dead end).
  - `experiment_data_crossing.xlsx` -- crossing / head-on traffic.

## Reproducing the nominal experiment

```bash
cd code
python benchmark.py --runs 100 --out experiment_data_v4.xlsx
```

Requirements: Python 3, NumPy, SciPy, and openpyxl.

The nominal result is an **82%** success rate for the complete framework versus **70%** for a plain A*+DWA baseline (100 paired trials, McNemar `p = 0.0075`).

## Citation

Citation information will be added once the manuscript is published.
