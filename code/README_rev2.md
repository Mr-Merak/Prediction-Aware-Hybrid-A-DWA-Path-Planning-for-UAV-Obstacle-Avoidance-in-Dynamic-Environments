# Round-2 experiment scripts

All commands assume the current directory is `code/` and use the same
interpreter that runs the original benchmark (`py` or the full python path).

## 1. Sigmoid vs linear vs threshold (Review 2, comment 2)

```bash
py alpha_modes.py --runs 100 --jobs 7
```

Output: `../data/experiment_data_alpha_modes.xlsx` (summary, paired tests,
raw runs). The manuscript reports the paired differences relative to the
sigmoid (Table 6); the absolute rates are in Supplementary Table S2.

## 2. 2x2 factorial on the manoeuvring scene (Review 2, comment 3)

```bash
py factorial_maneuver.py --runs 100 --jobs 7
```

Output: `../data/experiment_data_factorial_maneuver_crossing.xlsx` (the script
also runs the crossing scene; stop it after the manoeuvring block if only the
reviewer's minimum request is needed). The manoeuvring cells and simple effects
are in Supplementary Table S1 and Section 5.4.

## 3. 100-trial sensitivity sweeps (Review 2, comment 7)

```bash
py sensitivity100.py --runs 100 --jobs 7
py sensitivity_table.py                  # prints LaTeX rows
py update_sensitivity_table.py           # rewrites Table 12 in v8_paper.tex
py update_supplementary_sensitivity.py   # fills Supplementary Table S3
```

Output: `../data/experiment_data_sensitivity100.xlsx`,
`../data/sensitivity_latex_rows.txt`.

## 4. 6-DOF 3D validation (Expert 1; Review 2, comment 5)

```bash
py benchmark3d6dof.py --runs 100 --jobs 7
py make_figure_3d6dof.py                 # bar chart -> ../figure3d_6dof.png
```

Output: `../data/experiment_data_3d6dof.xlsx`,
`../data/trajectories_3d6dof.json`. The model is a rigid-body quadrotor with a
cascaded velocity/attitude controller; the two scenarios are an altitude
window with 3D movers and a dense 3D crossing scene. The true Euclidean 3D A*
baseline is the `astar3d` configuration; `ia3d` is the risk-adaptive planner.

## 5. Structured-scene outcome breakdown (Review 2, comment 6)

```bash
py structured_failure_table.py
```

Reads the existing `experiment_data_structured.xlsx`; writes
`../data/experiment_data_structured_failure.xlsx`. No re-run is needed.

## Notes

- `benchmark.py` now has an `ALPHA_MODE` global (`sigmoid`, `linear`,
  `threshold`); it is set per worker by `alpha_modes.py`. The default
  (`sigmoid`) reproduces the original heuristic.
- `benchmark3d.py` gained a `risk_adaptive` flag on `AStar3D` so the Euclidean
  3D baseline is a true A*, not the risk-adaptive planner. The default
  (`True`) preserves the original behaviour.
- All absolute rates in the revised manuscript are from the current
  implementation (the round-2 rerun workbooks); the earlier 2--5 percentage
  point discrepancy between the submitted tables and the code has been
  eliminated.
- The 3D A* priority queue was corrected in the round-3 review: the heuristic
  is now evaluated at the neighbour, not at its parent, so the Euclidean and
  risk-adaptive 3D planners are genuine A* searches. The 6-DOF results were
  regenerated with this fix and with a 1.6 m/s pure-pursuit speed cap for the
  global-only baselines.
