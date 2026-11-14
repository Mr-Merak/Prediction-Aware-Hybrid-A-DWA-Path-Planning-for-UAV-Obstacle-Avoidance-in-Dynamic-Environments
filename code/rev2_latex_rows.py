"""
rev2_latex_rows.py -- print LaTeX rows for the rerun tables.

Run after the rerun workbooks exist:
    py rev2_latex_rows.py
"""
import math
import os
import sys

from rev2_numbers import load, DATA


def get(rows, label):
    for r in rows:
        if r.get("Configuration") == label:
            return r
    return None


def num(v, fmt="%.1f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return fmt % v


def table3(path):
    summary, _ = load(path)
    order = ["Traditional A*", "D* Lite", "DWA-only", "A* + DWA", "IA*-only",
             "Proposed algorithm", "Proposed + Pred-DWA"]
    print("% Table 3")
    for label in order:
        r = get(summary, label)
        if r is None:
            continue
        succ = "%.1f\\ [%.1f,\\,%.1f]" % (r["Success Rate (%)"], r["95% CI Low"], r["95% CI High"])
        if label == "Proposed + Pred-DWA":
            print("\\textbf{%s} & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} & \\textbf{%s}\\\\" % (
                label, succ, num(r.get("Path Length (m)")), num(r.get("Tracking Error (m)"), "%.2f"),
                num(r.get("Replanning Time (s)"), "%.3f")))
        else:
            print("%s & %s & %s & %s & %s\\\\" % (
                label, succ, num(r.get("Path Length (m)")),
                num(r.get("Tracking Error (m)"), "%.2f"),
                num(r.get("Replanning Time (s)"), "%.3f")))
    # D* Lite successful count for the caption
    r = get(summary, "D* Lite")
    if r:
        print("% D* Lite successes:", r.get("Successes"), "of", r.get("Runs"))


def table4(path):
    summary, _ = load(path)
    mapping = [
        ("Proposed + Pred-DWA", "Complete framework (prediction-aware DWA)"),
        ("Proposed algorithm", "\\quad w/o prediction-aware DWA"),
        ("Ablation 1 (w/o KF)", "\\quad w/o Kalman prediction (CV extrapolation)"),
        ("Ablation 2 (w/o AdaH)", "\\quad w/o adaptive heuristic"),
        ("IA*-only", "\\quad w/o DWA (pure pursuit)"),
    ]
    print("\n% Table 4")
    for key, label in mapping:
        r = get(summary, key)
        if r is None:
            continue
        print("%s & %.1f\\ [%.1f,\\,%.1f] & %s & %s\\\\" % (
            label, r["Success Rate (%)"], r["95% CI Low"], r["95% CI High"],
            num(r.get("Path Length (m)")), num(r.get("Tracking Error (m)"), "%.2f")))


def table7(path):
    summary, _ = load(path)
    order = ["Traditional A*", "DWA-only", "IA*-only", "A* + DWA",
             "Proposed algorithm", "Proposed + Pred-DWA"]
    print("\n% Table 7")
    for label in order:
        r = get(summary, label)
        if r is None:
            continue
        succ = ("%.1f\\ [%.1f,\\,%.1f]" % (r["Success Rate (%)"], r["95% CI Low"], r["95% CI High"])
                if r.get("Runs") else "0.0")
        print("%s & %s & %s\\\\" % (
            label, succ, num(r.get("Tracking Error (m)"), "%.2f")))


def table8(path):
    summary, _ = load(path)
    order = ["Traditional A*", "IA*-only", "A* + DWA", "Proposed algorithm",
             "Proposed + Pred-DWA"]
    print("\n% Table 8")
    for label in order:
        r = get(summary, label)
        if r is None:
            continue
        print("%s & %.1f\\ [%.1f,\\,%.1f]\\\\" % (
            label, r["Success Rate (%)"], r["95% CI Low"], r["95% CI High"]))


def table9(path):
    summary, _ = load(path)
    scen = {"5 dynamic (12 static)": "5", "10 dynamic (12 static)": "10",
            "15 dynamic (12 static)": "15", "20 dynamic (12 static)": "20",
            "10 dynamic (24 static)": "10 (24 static)"}
    cols = ["IA*-only", "A* + DWA", "Proposed algorithm", "Proposed + Pred-DWA"]
    print("\n% Table 9")
    for s_label, row_label in scen.items():
        vals = []
        for c in cols:
            r = next((x for x in summary
                      if x.get("Scenario") == s_label and x.get("Configuration") == c), None)
            vals.append("%.1f\\ [%.1f,\\,%.1f]" % (
                r["Success Rate (%)"], r["95% CI Low"], r["95% CI High"]) if r else "--")
        print("%s & %s\\\\" % (row_label, " & ".join(vals)))


def table13(path):
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["Collision categories"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()
    print("\n% Table 13")
    for r in rows:
        cfg, runs, succ, head, cross, over, static, total = r[:8]
        print("%s & %d & %d & %d & %d & %d\\\\" % (cfg, succ, head, cross, over, static))


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    nom = os.path.join(DATA, "experiment_data_nominal_rerun.xlsx")
    if os.path.exists(nom):
        table3(nom)
        table4(nom)
    for fname, fn in [("experiment_data_maneuver_rerun.xlsx", table7),
                      ("experiment_data_crossing.xlsx", table8),
                      ("experiment_data_density.xlsx", table9),
                      ("experiment_data_failure.xlsx", table13)]:
        p = os.path.join(DATA, fname)
        if os.path.exists(p):
            fn(p)


if __name__ == "__main__":
    main()
