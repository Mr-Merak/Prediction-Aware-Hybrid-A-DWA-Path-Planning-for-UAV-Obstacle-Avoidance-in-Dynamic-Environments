"""
update_supplementary_sensitivity.py -- insert the full sensitivity rows into
Supplementary Table S3.

Reads data/sensitivity_latex_rows.txt (written by update_sensitivity_table.py)
and replaces the '% ROWS' placeholder in supplementary_material.tex.

Run:
    py update_supplementary_sensitivity.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    rows_path = os.path.join(ROOT, "data", "sensitivity_latex_rows.txt")
    tex_path = os.path.join(ROOT, "supplementary_material.tex")

    lines = open(rows_path, encoding="utf-8").read().splitlines()
    marker = "% Full rows (rate [CI low, CI high])"
    idx = lines.index(marker)
    rows = [ln for ln in lines[idx + 1:] if ln.strip()]

    tex = open(tex_path, encoding="utf-8").read()
    if "% ROWS" not in tex:
        raise SystemExit("placeholder % ROWS not found")
    tex = tex.replace("% ROWS", "\n".join(rows))
    open(tex_path, "w", encoding="utf-8").write(tex)
    print("Inserted %d rows into %s" % (len(rows), tex_path))


if __name__ == "__main__":
    main()
