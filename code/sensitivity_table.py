"""
sensitivity_table.py -- render the 100-trial sensitivity results as LaTeX rows.

Reads data/experiment_data_sensitivity100.xlsx and prints:
  * a compact table (success rate only), and
  * a full table (success rate with Wilson 95% CI in brackets),
for pasting into Section 5.8 / Supplementary Material.

Run:
    py sensitivity_table.py
"""
import math
import os
import sys

from openpyxl import load_workbook

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data",
                    "experiment_data_sensitivity100.xlsx")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    wb = load_workbook(DATA, read_only=True, data_only=True)
    ws = wb["Sensitivity 100"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()

    # rows: Parameter, Value, Runs, Successes, Rate, CI low, CI high, ...
    groups = {}
    order = []
    for r in rows:
        p = r[0]
        if p not in groups:
            groups[p] = []
            order.append(p)
        groups[p].append(r)

    print("% compact rows (rate only)")
    for p in order:
        cells = " & ".join("%s: %.1f" % (r[1], r[4]) for r in groups[p])
        print("%s & %s \\\\" % (p, cells))

    print("\n% full rows (rate [CI low, CI high])")
    for p in order:
        cells = " & ".join("%s: %.1f [%.1f, %.1f]" % (r[1], r[4], r[5], r[6])
                           for r in groups[p])
        print("%s & %s \\\\" % (p, cells))

    print("\n% summary")
    rates = [r[4] for r in rows]
    print("%% overall range: %.1f--%.1f%% across %d settings"
          % (min(rates), max(rates), len(rates)))
    for p in order:
        rs = groups[p]
        print("%% %-30s min %.1f (%s), max %.1f (%s)"
              % (p, min(r[4] for r in rs), min(rs, key=lambda x: x[4])[1],
                 max(r[4] for r in rs), max(rs, key=lambda x: x[4])[1]))


if __name__ == "__main__":
    main()
