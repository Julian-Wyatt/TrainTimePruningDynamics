#!/usr/bin/env python3
"""Build throughput-vs-quality curve CSVs for the pruning study figures.

Joins per-config throughput (results/throughput.csv) with test-set quality
metrics (qA_keep_sweep / qA_core / qBC_cropr_vits) by ``config_leaf`` and emits
one CSV per series into results/graph_data/, ready for pgfplots
``table [x=throughput, y=<metric>, col sep=comma]``.

Notes
-----
* qA keep-sweep is seed-42 only; its r07 centre is pulled from qA_core
  (gumbel_route_penultimate / soft_mask) at seed 42 for consistency.
* qBC is averaged over seeds 42/43/44 (mean + sample std). Its step3 r07 centre
  is aux_route_penultimate in qA_core (== 3-stage, keep 0.7, penultimate).
* per_block budget variants (t256/288/384) are throughput-only, so per_block is
  a single accuracy point (the derived ~352 tok/block rate).
* Throughput here is the throughput_mode (full-attention, single-mask) operating
  point -- a *relative* x-axis, identical across all rows (see paper caveat).
"""
import csv
import os
import statistics

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RES, "graph_data")
os.makedirs(OUT, exist_ok=True)

METRICS = [
    ("cldice", "test_cldice"),
    ("miou", "test_mIoU"),
    ("thin_recall", "test_thin_vessel_recall"),
    ("boundary_iou", "test_boundary_iou"),
]


def load(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


# config_leaf -> (img/s, std)
thr = {}
for r in load("throughput.csv"):
    thr[r["config_leaf"]] = (
        float(r["throughput_img_per_sec"]),
        float(r["throughput_img_per_sec_std"]),
    )

ks = load("qA_keep_sweep.csv")
core = load("qA_core.csv")
bc = load("qBC_cropr_vits.csv")


def rows_for(rows, leaf, seed=None):
    out = [r for r in rows if r["config_leaf"] == leaf]
    if seed is not None:
        out = [r for r in out if r["seed"] == str(seed)]
    return out


def agg(rows):
    res = {}
    for name, col in METRICS:
        vals = [float(r[col]) for r in rows]
        m = sum(vals) / len(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        res[name] = (m, sd)
    return res


def make_point(keep_label, thr_leaf, acc_rows):
    if not acc_rows:
        raise SystemExit(f"no accuracy rows for throughput leaf {thr_leaf!r}")
    if thr_leaf not in thr:
        raise SystemExit(f"no throughput row for leaf {thr_leaf!r}")
    ips, ips_sd = thr[thr_leaf]
    p = {"keep": keep_label, "throughput": ips, "throughput_sd": ips_sd,
         "n": len(acc_rows)}
    p.update(agg(acc_rows))
    return p


def write_series(fname, points):
    cols = ["keep", "throughput", "throughput_sd"]
    for name, _ in METRICS:
        cols += [name, name + "_sd"]
    points = sorted(points, key=lambda p: p["throughput"])
    with open(os.path.join(OUT, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in points:
            row = [p["keep"], f'{p["throughput"]:.3f}', f'{p["throughput_sd"]:.4f}']
            for name, _ in METRICS:
                m, sd = p[name]
                row += [f"{m:.5f}", f"{sd:.5f}"]
            w.writerow(row)
    return points


# ---- qA keep sweep (seed 42) ----
def qa_series(prefix, r07_leaf):
    pts = []
    for ri in (5, 6, 7, 8, 9):
        if ri == 7:
            acc = rows_for(core, r07_leaf, seed=42)
            tleaf = f"{prefix}_r07"
        else:
            acc = rows_for(ks, f"{prefix}_r0{ri}", seed=42)
            tleaf = f"{prefix}_r0{ri}"
        pts.append(make_point(f"0.{ri}", tleaf, acc))
    return pts


gum = write_series("qA_keep_gumbel.csv", qa_series("gumbel_route", "gumbel_route_penultimate"))
sm = write_series("qA_keep_softmask.csv", qa_series("soft_mask", "soft_mask"))

# ---- qBC jump1 (3-seed mean) ----
jp = write_series("qBC_jump1.csv",
                  [make_point(f"0.{ri}", f"jump1_r0{ri}", rows_for(bc, f"jump1_r0{ri}"))
                   for ri in (2, 3, 4, 5, 6, 7, 8, 9)])

# ---- qBC step3 (3-seed mean); r07 centre from qA_core ----
st_pts = []
for ri in (5, 6, 7, 8, 9):
    if ri == 7:
        st_pts.append(make_point("0.7", "step3_r07", rows_for(core, "aux_route_penultimate")))
    else:
        st_pts.append(make_point(f"0.{ri}", f"step3_r0{ri}", rows_for(bc, f"step3_r0{ri}")))
st = write_series("qBC_step3.csv", st_pts)

# ---- qBC per_block budget sweep (3-seed mean); t352 == derived per_block ----
pb = write_series("qBC_per_block.csv",
                  [make_point(str(t), f"per_block_t{t}", rows_for(bc, f"per_block_t{t}"))
                   for t in (256, 288, 352, 384)])
dn = write_series("dense_ref.csv", [make_point("1.0", "dense", rows_for(core, "dense"))])

# ---- console summary ----
def show(name, pts):
    print(f"\n{name}")
    print(f'  {"keep":>6} {"img/s":>9} {"cldice":>8} {"mIoU":>8} {"thinR":>8} {"bIoU":>8}  n')
    for p in pts:
        print(f'  {p["keep"]:>6} {p["throughput"]:>9.1f} '
              f'{p["cldice"][0]:>8.4f} {p["miou"][0]:>8.4f} '
              f'{p["thin_recall"][0]:>8.4f} {p["boundary_iou"][0]:>8.4f}  {p["n"]}')


for nm, pts in [("qA gumbel", gum), ("qA soft_mask", sm),
                ("qBC jump1", jp), ("qBC step3", st),
                ("qBC per_block", pb), ("dense", dn)]:
    show(nm, pts)
print(f"\nwrote CSVs to {os.path.normpath(OUT)}")
