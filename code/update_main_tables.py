"""
update_main_tables.py -- rewrite the main result tables from the rerun workbooks.

Rebuilds the row blocks of Tables 3, 4, 5 (nominal row), 7, 8, 9 and 13 in
v8_paper.tex from the rerun xlsx files. Captions/prose are left intact.

Run after the rerun workbooks exist:
    py update_main_tables.py
"""
import math
import os
import re
import sys

from openpyxl import load_workbook

from rev2_numbers import load, DATA

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "v8_paper.tex")


def _num(v, fmt="%.1f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return fmt % v


def _succ(r):
    return "%.1f\\ [%.1f,\\,%.1f]" % (r["Success Rate (%)"], r["95% CI Low"], r["95% CI High"])


def _get(summary, label):
    for r in summary:
        if r.get("Configuration") == label:
            return r
    return None


def replace_rows(tex, label, rows):
    lab = tex.index("\\label{%s}" % label)
    start = tex.rindex("\\begin{table}", 0, lab)
    end = tex.index("\\end{table}", lab) + len("\\end{table}")
    block = tex[start:end]
    a = block.index("\\midrule") + len("\\midrule")
    b = block.index("\\bottomrule")
    block = block[:a] + "\n" + rows.strip() + "\n" + block[b:]
    return tex[:start] + block + tex[end:]


def table3_rows(path):
    summary, _ = load(path)
    order = ["Traditional A*", "D* Lite", "DWA-only", "A* + DWA", "IA*-only",
             "Proposed algorithm", "Proposed + Pred-DWA"]
    out = []
    for label in order:
        r = _get(summary, label)
        if r is None:
            continue
        if label == "Proposed + Pred-DWA":
            out.append("\\textbf{%s} & \\textbf{%s} & \\textbf{%s} & \\textbf{%s} & \\textbf{%s}\\\\" % (
                label, _succ(r), _num(r.get("Path Length (m)")),
                _num(r.get("Tracking Error (m)"), "%.2f"),
                _num(r.get("Replanning Time (s)"), "%.3f")))
        else:
            out.append("%s & %s & %s & %s & %s\\\\" % (
                label, _succ(r), _num(r.get("Path Length (m)")),
                _num(r.get("Tracking Error (m)"), "%.2f"),
                _num(r.get("Replanning Time (s)"), "%.3f")))
    return "\n".join(out)


def table4_rows(path):
    summary, _ = load(path)
    mapping = [
        ("Proposed + Pred-DWA", "Complete framework (prediction-aware DWA)"),
        ("Proposed algorithm", "\\quad w/o prediction-aware DWA"),
        ("Ablation 1 (w/o KF)", "\\quad w/o Kalman prediction (CV extrapolation)"),
        ("Ablation 2 (w/o AdaH)", "\\quad w/o adaptive heuristic"),
        ("IA*-only", "\\quad w/o DWA (pure pursuit)"),
    ]
    out = []
    for key, label in mapping:
        r = _get(summary, key)
        if r is None:
            continue
        out.append("%s & %s & %s & %s\\\\" % (
            label, _succ(r), _num(r.get("Path Length (m)")),
            _num(r.get("Tracking Error (m)"), "%.2f")))
    return "\n".join(out)


def table5_nominal(tex, path):
    summary, _ = load(path)
    cells = {}
    for key in ["A* + DWA", "A* + Pred-DWA", "Proposed algorithm", "Proposed + Pred-DWA"]:
        r = _get(summary, key)
        cells[key] = r
    # The factorial table uses rates without CIs (it has its own CIs from the
    # factorial script in the original; keep the CI format consistent).
    def cell(r):
        return "%.0f\\ [%.1f,\\,%.1f]" % (r["Success Rate (%)"], r["95% CI Low"], r["95% CI High"])
    row = ("Nominal & %s & %s & %s & \\textbf{%s}\\\\" % (
        cell(cells["A* + DWA"]), cell(cells["A* + Pred-DWA"]),
        cell(cells["Proposed algorithm"]), cell(cells["Proposed + Pred-DWA"])))
    return re.sub(r"^Nominal &.*$", lambda m: row, tex, count=1, flags=re.M)


def table7_rows(path):
    summary, _ = load(path)
    order = ["Traditional A*", "DWA-only", "IA*-only", "A* + DWA",
             "Proposed algorithm", "Proposed + Pred-DWA"]
    out = []
    for label in order:
        r = _get(summary, label)
        if r is None:
            continue
        succ = _succ(r) if r.get("Runs") else "0.0"
        out.append("%s & %s & %s\\\\" % (
            label, succ, _num(r.get("Tracking Error (m)"), "%.2f")))
    return "\n".join(out)


def table8_rows(path):
    summary, _ = load(path)
    order = ["Traditional A*", "IA*-only", "A* + DWA", "Proposed algorithm",
             "Proposed + Pred-DWA"]
    out = []
    for label in order:
        r = _get(summary, label)
        if r is None:
            continue
        out.append("%s & %s\\\\" % (label, _succ(r)))
    return "\n".join(out)


def table9_rows(path):
    summary, _ = load(path)
    scen = {"5 dynamic (12 static)": "5", "10 dynamic (12 static)": "10",
            "15 dynamic (12 static)": "15", "20 dynamic (12 static)": "20",
            "10 dynamic (24 static)": "10 (24 static)"}
    cols = ["IA*-only", "A* + DWA", "Proposed algorithm", "Proposed + Pred-DWA"]
    out = []
    for s_label, row_label in scen.items():
        vals = []
        for c in cols:
            r = next((x for x in summary
                      if x.get("Scenario") == s_label and x.get("Configuration") == c), None)
            vals.append(_succ(r) if r else "--")
        out.append("%s & %s\\\\" % (row_label, " & ".join(vals)))
    return "\n".join(out)


def table13_rows(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["Collision categories"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()
    out = []
    for r in rows:
        cfg, runs, succ, head, cross, over, static, total = r[:8]
        out.append("%s & %d & %d & %d & %d & %d\\\\" % (cfg, succ, head, cross, over, static))
    return "\n".join(out)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    tex = open(TEX, encoding="utf-8").read()

    nom = os.path.join(DATA, "experiment_data_nominal_rerun.xlsx")
    if os.path.exists(nom):
        tex = replace_rows(tex, "tab:comparison", table3_rows(nom))
        tex = replace_rows(tex, "tab:ablation", table4_rows(nom))
        tex = table5_nominal(tex, nom)
        print("Updated Tables 3, 4, 5 (nominal).")

    for fname, label, fn in [
            ("experiment_data_maneuver_rerun.xlsx", "tab:maneuver", table7_rows),
            ("experiment_data_crossing.xlsx", "tab:crossing", table8_rows),
            ("experiment_data_density.xlsx", "tab:density", table9_rows),
            ("experiment_data_failure.xlsx", "tab:failure", table13_rows)]:
        p = os.path.join(DATA, fname)
        if os.path.exists(p):
            tex = replace_rows(tex, label, fn(p))
            print("Updated %s." % label)

    open(TEX, "w", encoding="utf-8").write(tex)


if __name__ == "__main__":
    main()
