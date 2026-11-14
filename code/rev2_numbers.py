"""
rev2_numbers.py -- summarise the re-run tables for the round-2 manuscript.

Reads the rerun workbooks and prints compact numbers for Tables 3, 4, 5, 7,
8, 9, 13 and the abstract/conclusions. McNemar tests are paired by seed.

Run:
    py rev2_numbers.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
from openpyxl import load_workbook
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")


def load(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    summary = []
    ws = wb["Summary"]
    hdr = [c for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    for r in ws.iter_rows(min_row=2, values_only=True):
        summary.append(dict(zip(hdr, r)))
    raw = []
    if "Raw runs" in wb.sheetnames:
        ws = wb["Raw runs"]
        hdr = [c for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
        for r in ws.iter_rows(min_row=2, values_only=True):
            raw.append(dict(zip(hdr, r)))
    wb.close()
    return summary, raw


def mcnemar(raw, cfg_a, cfg_b):
    piv = defaultdict(dict)
    for r in raw:
        if r.get("Configuration") in (cfg_a, cfg_b):
            piv[r["Configuration"]][r["Seed"]] = int(r["Success"])
    seeds = sorted(set(piv[cfg_a]) & set(piv[cfg_b]))
    a = np.array([piv[cfg_a][s] for s in seeds])
    b = np.array([piv[cfg_b][s] for s in seeds])
    bc = int(np.sum((a == 1) & (b == 0)))
    cb = int(np.sum((a == 0) & (b == 1)))
    p = binomtest(min(bc, cb), bc + cb, 0.5).pvalue if bc + cb else float("nan")
    return bc, cb, p


def print_summary(path, title):
    summary, raw = load(path)
    print("\n===== %s =====" % title)
    print("%-48s %6s %-16s %8s %8s %8s" % (
        "Configuration", "Succ%", "95% CI", "Path", "Track", "Replan"))
    for s in summary:
        if s.get("Runs", 0) in (0, None):
            continue
        print("%-48s %5.1f  [%4.1f,%4.1f] %8.1f %8.2f %8.4f" % (
            s["Configuration"], s["Success Rate (%)"],
            s["95% CI Low"], s["95% CI High"],
            s.get("Path Length (m)") or float("nan"),
            s.get("Tracking Error (m)") or float("nan"),
            s.get("Replanning Time (s)") or float("nan")))
    return summary, raw


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    nom_path = os.path.join(DATA, "experiment_data_nominal_rerun.xlsx")
    if os.path.exists(nom_path):
        summary, raw = print_summary(nom_path, "Nominal (Table 3 / 4 / 5)")
        print("\nPaired McNemar:")
        for a, b in [("Proposed + Pred-DWA", "A* + DWA"),
                     ("Proposed + Pred-DWA", "Proposed algorithm"),
                     ("Proposed algorithm", "A* + DWA"),
                     ("Proposed + Pred-DWA", "Ablation 1 (w/o KF)"),
                     ("Proposed + Pred-DWA", "Ablation 2 (w/o AdaH)"),
                     ("Proposed + Pred-DWA", "IA*-only")]:
            bc, cb, p = mcnemar(raw, a, b)
            print("  %-34s vs %-30s b/c=%d/%d p=%.4f" % (a, b, bc, cb, p))
        # factorial nominal
        import factorial_ablation as FA
        mapping = {"A* + DWA": "astar_dwa", "A* + Pred-DWA": "astar_pred_dwa",
                   "Proposed algorithm": "proposed",
                   "Proposed + Pred-DWA": "proposed_pred_dwa"}
        rows = []
        for r in raw:
            if r["Configuration"] in mapping:
                rows.append(dict(config=mapping[r["Configuration"]],
                                 seed=r["Seed"], success=int(r["Success"])))
        ft = FA.factorial_table(rows)
        FA.print_table("nominal", "Nominal re-run", ft)

    man_path = os.path.join(DATA, "experiment_data_maneuver_rerun.xlsx")
    if os.path.exists(man_path):
        summary, raw = print_summary(man_path, "Maneuver (Table 7)")
        import factorial_ablation as FA
        mapping = {"A* + DWA": "astar_dwa", "A* + Pred-DWA": "astar_pred_dwa",
                   "Proposed algorithm": "proposed",
                   "Proposed + Pred-DWA": "proposed_pred_dwa"}
        rows = []
        for r in raw:
            if r["Configuration"] in mapping:
                rows.append(dict(config=mapping[r["Configuration"]],
                                 seed=r["Seed"], success=int(r["Success"])))
        ft = FA.factorial_table(rows)
        FA.print_table("maneuver", "Maneuver re-run", ft)

    for fname, title in [
            ("experiment_data_crossing.xlsx", "Crossing (Table 8)"),
            ("experiment_data_density.xlsx", "Density (Table 9)")]:
        path = os.path.join(DATA, fname)
        if os.path.exists(path):
            print_summary(path, title)

    # Failure workbook uses a "Collision categories" sheet; print it separately.
    fp = os.path.join(DATA, "experiment_data_failure.xlsx")
    if os.path.exists(fp):
        from openpyxl import load_workbook
        wb = load_workbook(fp, read_only=True, data_only=True)
        ws = wb["Collision categories"]
        print("\n===== Failure (Table 13) =====")
        for r in ws.iter_rows(min_row=1, values_only=True):
            print(r)
        wb.close()


if __name__ == "__main__":
    main()
