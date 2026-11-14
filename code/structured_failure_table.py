"""
structured_failure_table.py -- outcome breakdown for the structured scenes.

Review 2, round 2, comment #6: the failure-mode analysis (Table 11) covers only
collisions on the nominal scene, while the structured scenes also contain
non-collision failures (the reactive DWA oscillates in the gap and the run
exhausts the time limit).  This script turns the existing
experiment_data_structured.xlsx summary into an outcome table with a
"stuck / oscillation" category, so the taxonomy question can be answered
without re-running anything.

Note: in the stored data these episodes are recorded as `timeout` (the UAV
never reaches the goal); the paper describes the same episodes as "stuck".
They are grouped here as `stuck / oscillation (time limit)`.

Run:
    py structured_failure_table.py
"""
import math
import os
import sys

from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, Alignment

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

CONFIGS = ["A* + DWA", "Proposed algorithm", "Proposed + Pred-DWA"]
SCENES = ["Narrow passage", "Bottleneck", "Dead-end (concave U)"]


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    wb = load_workbook(os.path.join(DATA_DIR, "experiment_data_structured.xlsx"),
                       read_only=True, data_only=True)
    ws = wb["Summary"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()

    by_key = {(r[0], r[1]): r for r in rows}

    out = []
    for scene in SCENES:
        for cfg in CONFIGS:
            r = by_key.get((scene, cfg))
            if r is None:
                continue
            _, _, runs, rate, lo, hi, path, track, step, coll, tout, stuck = r[:12]
            noncollision = (tout or 0) + (stuck or 0)
            out.append(dict(scene=scene, cfg=cfg, runs=runs, rate=rate,
                            lo=lo, hi=hi, path=path, track=track, step=step,
                            coll=coll or 0, timeout=tout or 0, stuck=stuck or 0,
                            oscillation=noncollision))

    print("\n===== Structured-scene outcome breakdown (100 paired trials each) =====")
    print("%-22s %-22s %7s %-15s %10s %14s" % (
        "Scene", "Configuration", "Succ%", "95% CI", "Collision", "Stuck/oscill."))
    for r in out:
        print("%-22s %-22s %6.1f  [%4.1f,%4.1f] %10d %14d" % (
            r["scene"], r["cfg"], r["rate"], r["lo"], r["hi"],
            r["coll"], r["oscillation"]))

    # LaTeX rows for the manuscript table
    print("\n% LaTeX rows (Success / Collision / Stuck-oscillation):")
    for r in out:
        print("%-22s & %-22s & %d & %d & %d \\\\" % (
            r["scene"], r["cfg"], r["rate"], r["coll"], r["oscillation"]))

    # xlsx
    path = os.path.join(DATA_DIR, "experiment_data_structured_failure.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "Outcomes"
    ws.append(["Scenario", "Configuration", "Runs", "Success Rate (%)",
               "95% CI Low", "95% CI High", "Collisions",
               "Stuck / oscillation (recorded timeout)", "Path Length (m)"])
    for c in ws[1]:
        c.font = Font(bold=True)
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for r in out:
        ws.append([r["scene"], r["cfg"], r["runs"], r["rate"], r["lo"], r["hi"],
                   r["coll"], r["oscillation"],
                   r["path"] if r["path"] is not None else None])
    for col in "ABCDEFGHI":
        ws.column_dimensions[col].width = 22
    wb.save(path)
    print("\nWrote %s" % path)


if __name__ == "__main__":
    main()
