#!/usr/bin/env python3
"""Collate a results table into a per-experiment mean±std summary.

Reads CSV tables written by ``src/utils/results_registry.py`` (one row per
``(experiment_id, seed)``) and aggregates the requested metrics across seeds,
so a seed sweep becomes a single paper-ready table without collating W&B by hand.

Examples:
  python scripts/collate_results.py --table saves/results/qA_core.csv
  python scripts/collate_results.py --dir saves/results --all --latex \
      --metrics mIoU,dice_fg,cldice,boundary_iou,boundary_ap
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
from collections import OrderedDict
from typing import Dict, List


def _load_rows(paths: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _resolve_column(row: Dict[str, str], metric: str) -> str | None:
    """A requested metric name maps to its `test_<name>` column (or itself)."""
    for candidate in (f"test_{metric}", metric):
        if candidate in row:
            return candidate
    return None


def _aggregate(rows: List[Dict[str, str]], metrics: List[str]):
    by_exp: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
    for row in rows:
        by_exp.setdefault(row.get("experiment_id", "?"), []).append(row)

    table = []
    for exp_id, exp_rows in by_exp.items():
        seeds = sorted({str(r.get("seed", "")) for r in exp_rows if r.get("seed") != ""})
        entry = {"experiment_id": exp_id, "n": len(seeds), "seeds": ",".join(seeds)}
        for metric in metrics:
            values: List[float] = []
            for row in exp_rows:
                col = _resolve_column(row, metric)
                if col and row.get(col, "") not in ("", None):
                    try:
                        values.append(float(row[col]))
                    except ValueError:
                        pass
            if values:
                mean = statistics.fmean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                entry[metric] = (mean, std)
            else:
                entry[metric] = None
        table.append(entry)
    return table


def _fmt_cell(cell) -> str:
    if cell is None:
        return "—"
    mean, std = cell
    return f"{mean:.4f} ± {std:.4f}"


def _render_markdown(table, metrics: List[str]) -> str:
    head = ["experiment_id", "n"] + metrics
    lines = ["| " + " | ".join(head) + " |",
             "| " + " | ".join(["---"] * len(head)) + " |"]
    for entry in table:
        cells = [entry["experiment_id"], str(entry["n"])] + [
            _fmt_cell(entry[m]) for m in metrics
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_latex(table, metrics: List[str]) -> str:
    cols = "l r " + " ".join(["r"] * len(metrics))
    header = " & ".join(["experiment\\_id", "n"] + [m.replace("_", "\\_") for m in metrics])
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", header + " \\\\", "\\midrule"]
    for entry in table:
        cells = [entry["experiment_id"].replace("_", "\\_"), str(entry["n"])]
        for m in metrics:
            cell = entry[m]
            cells.append("--" if cell is None else f"${cell[0]:.4f} \\pm {cell[1]:.4f}$")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--table", help="path to a single group CSV")
    src.add_argument("--all", action="store_true",
                     help="aggregate every *.csv under --dir")
    ap.add_argument("--dir", default="saves/results",
                    help="results dir for --all (default: saves/results)")
    ap.add_argument("--metrics", default="mIoU,dice_fg,cldice,boundary_iou",
                    help="comma-separated metric names (matched to test_<name>)")
    ap.add_argument("--sort-by", default=None,
                    help="metric to sort rows by, descending")
    ap.add_argument("--latex", action="store_true", help="emit a LaTeX tabular")
    args = ap.parse_args()

    if args.all:
        paths = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
        if not paths:
            ap.error(f"no CSVs found under {args.dir}")
    else:
        if not os.path.exists(args.table):
            ap.error(f"table not found: {args.table}")
        paths = [args.table]

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    rows = _load_rows(paths)
    if not rows:
        ap.error("no rows to collate")
    table = _aggregate(rows, metrics)

    if args.sort_by and args.sort_by in metrics:
        table.sort(key=lambda e: (e[args.sort_by] or (-1e9, 0))[0], reverse=True)

    print(_render_latex(table, metrics) if args.latex
          else _render_markdown(table, metrics))


if __name__ == "__main__":
    main()
