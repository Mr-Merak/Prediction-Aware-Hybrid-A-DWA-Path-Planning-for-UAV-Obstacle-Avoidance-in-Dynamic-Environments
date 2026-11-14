"""
update_sensitivity_table.py -- replace the sensitivity table in the manuscript
with the 100-trial results, and emit rows for the supplementary full-CI table.

Reads data/experiment_data_sensitivity100.xlsx (written by sensitivity100.py),
rewrites the table block that carries \\label{tab:sensitivity} in v8_paper.tex,
and writes data/sensitivity_latex_rows.txt with compact and full rows.

Run:
    py update_sensitivity_table.py
"""
import os
import re
import sys

from openpyxl import load_workbook

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
XLSX = os.path.join(ROOT, "data", "experiment_data_sensitivity100.xlsx")
TEX = os.path.join(ROOT, "v8_paper.tex")
OUT = os.path.join(ROOT, "data", "sensitivity_latex_rows.txt")

# nominal values to bold in the compact table
NOMINAL = {
    "prediction horizon (steps)": "25",
    "beta (heuristic steepness)": "10.0",
    "observation noise R": "2.0",
    "process noise Q": "0.5",
}
ORDER = ["prediction horizon (steps)", "beta (heuristic steepness)",
         "observation noise R", "process noise Q"]
ROW_LABEL = {
    "prediction horizon (steps)": "Prediction horizon (steps)",
    "beta (heuristic steepness)": r"$\beta$",
    "observation noise R": r"Observation noise $R$",
    "process noise Q": r"Process noise $Q$",
}
MAX_COLS = 5


def _fmt(v):
    if isinstance(v, float) and v.is_integer():
        return "%d" % int(v)
    return str(v)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    wb = load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["Sensitivity 100"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()

    groups = {p: [] for p in ORDER}
    for r in rows:
        groups[r[0]].append(r)

    compact_rows = []
    full_lines = []
    for p in ORDER:
        rs = groups[p]
        cells = []
        for r in rs:
            val = _fmt(r[1])
            rate = "%.1f" % r[4]
            cell = "%s: %s" % (val, rate)
            if NOMINAL.get(p) == val:
                cell = r"\textbf{%s}" % cell
            cells.append(cell)
        while len(cells) < MAX_COLS:
            cells.append("--")
        compact_rows.append("%s & %s \\\\" % (ROW_LABEL[p], " & ".join(cells)))
        for r in rs:
            full_lines.append(
                "%s & %s & %d & %.1f & [%.1f, %.1f] \\\\"
                % (ROW_LABEL[p], _fmt(r[1]), r[2], r[4], r[5], r[6]))

    compact = "\n".join(compact_rows)
    table = r"""\begin{table}[H]
\small
\caption{Parameter sensitivity of the proposed framework (100 paired trials per setting; the success rate is reported in \%; the nominal setting is marked in bold; Wilson 95\% confidence intervals for every point are reported in Supplementary Table~S3).\label{tab:sensitivity}}
%\isPreprints{\centering}{}
\begin{tabularx}{\textwidth}{lCCCCC}
\toprule
\textbf{Parameter} & \multicolumn{5}{c}{\textbf{Success rate (\%) at each value}}\\
\midrule
""" + compact + r"""
\bottomrule
\end{tabularx}
\end{table}"""

    tex = open(TEX, encoding="utf-8").read()
    # locate the table block carrying tab:sensitivity
    lab = tex.index(r"\label{tab:sensitivity}")
    start = tex.rindex(r"\begin{table}[H]", 0, lab)
    end = tex.index(r"\end{table}", lab) + len(r"\end{table}")
    tex = tex[:start] + table + tex[end:]
    open(TEX, "w", encoding="utf-8").write(tex)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("% Compact rows (rate only)\n")
        f.write(compact + "\n\n")
        f.write("% Full rows (rate [CI low, CI high])\n")
        f.write("\n".join(full_lines) + "\n")

    rates = [r[4] for r in rows]
    print("Replaced sensitivity table in %s" % TEX)
    print("Wrote %s" % OUT)
    print("Overall range: %.1f--%.1f%% over %d settings"
          % (min(rates), max(rates), len(rates)))
    for p in ORDER:
        rs = groups[p]
        lo = min(rs, key=lambda x: x[4])
        hi = max(rs, key=lambda x: x[4])
        print("  %-30s min %.1f at %s, max %.1f at %s"
              % (p, lo[4], _fmt(lo[1]), hi[4], _fmt(hi[1])))


if __name__ == "__main__":
    main()
