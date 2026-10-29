"""Paper figures: compression ratio against relative error (linear axes by default, --logx and
--logy for log axes), in the style of Kilmer, Horesh, Avron and Newman (PNAS 2021), Fig. 3.

    results/exp10_noise0.csv        quantum circuits, SVD over K = M_8(C) (curves in CURVES_EXP10)
    results/exp3_format.csv         Vlasov-Poisson, K = R[Z_32 x Z_32] (CURVES_EXP3)
    results/exp2_lon_format.csv     ERA5, K = R[Z_64] (CURVES_EXP2_LON)
    results/exp2_lonlat_format.csv  ERA5, K = R[Z_64] (x) DCT_32 (CURVES_EXP2_LONLAT)
    results/exp2_lon_ttn.csv        ERA5, tree tensor networks over K = R[Z_64] (CURVES_EXP2_TTN)
The curves are chosen by the name of the csv.

On a linear error axis, points with relative error below xmax / 400 are indistinguishable from 0;
of those only the one with the largest compression ratio is drawn (the rest would stack on the
y-axis).  Markers closer than --mark-gap (default 1.5%, 8% for exp2 and exp3) of the axes to the
previous marker are skipped; the lines still pass through every point.

Usage (the paper figures):
    python experiments/plot_paper.py exp10_noise0 --logx --logy --xmax 0.1 --ymax 2000 --mark-gap 0.04 \
        --out figures/exp10_noise0_paper_log
    python experiments/plot_paper.py exp3_format --xmax 0.5 --xstep 0.1 --logy --legend "lower right" --ncol 2 \
        --drop "TT over K" --out figures/exp3_format_paper_log
    python experiments/plot_paper.py exp2_lon_format --xmax 0.5 --xstep 0.1 --logy --out figures/exp2_lon_format_paper_log
    python experiments/plot_paper.py exp2_lonlat_format --xmax 0.5 --xstep 0.1 --logy --ymax 100 \
        --out figures/exp2_lonlat_format_paper_log
    python experiments/plot_paper.py exp2_lon_ttn --xmax 0.5 --xstep 0.1 --out figures/exp2_lon_ttn_paper
    python experiments/plot_paper.py exp2_lon_ttn --xmax 0.5 --xstep 0.1 --logy --out figures/exp2_lon_ttn_paper_log
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# colorblind-safe palette (Paul Tol, bright), indexed by the color index of each curve
PALETTE = ["#EE6677", "#4477AA", "#228833", "#AA3377", "#000000", "#EE7733", "#33BBEE"]

# every curve: (keys (method, truncation) in the csv, legend label, marker, color index, line style)
# quantum circuits (results/exp10_noise0.csv)
CURVES_EXP10 = [
    ((("SVD over K", "adaptive"),), r"SVD over $K$", "x", 0, "-"),
    ((("SVD over C of A_(1)", "uniform"),), r"SVD over $\mathbb{C}$ of $A_{(1)}$", ">", 2, "-"),
    ((("SVD over C of A_(2)", "uniform"),), r"SVD over $\mathbb{C}$ of $A_{(2)}$", "o", 3, "-"),
    ((("HOSVD over C of A", "uniform"),), r"HOSVD over $\mathbb{C}$ of $A$", "s", 4, "-"),
]

# Vlasov-Poisson (results/exp3_format.csv): one marker per format, solid over K, dashed over R, and a
# color of its own for every curve
XY, RR = "R[Z_32 x Z_32] on x and y", "R (no algebra)"
CURVES_EXP3 = [
    (((f"{XY}, TT", "adaptive"),), r"TT over $K$", "x", 0, "-"),
    (((f"{XY}, HOSVD", "adaptive"),), r"HOSVD over $K$", "s", 1, "-"),
    (((f"{XY}, ST-HOSVD", "adaptive"),), r"ST-HOSVD over $K$", "o", 2, "-"),
    (((f"{RR}, TT", "uniform"),), r"TT over $\mathbb{R}$", "+", 6, "--"),
    (((f"{RR}, HOSVD", "uniform"),), r"HOSVD over $\mathbb{R}$", "s", 4, "--"),
    (((f"{RR}, ST-HOSVD", "uniform"),), r"ST-HOSVD over $\mathbb{R}$", "o", 5, "--"),
    ((("matrix SVD (x, y) | (vx, vy, t)", "rank"),), r"SVD over $\mathbb{R}$ of $F_{(x,y)}$", "<", 3, "-."),
]


# ERA5, two experiments with the colors and markers of CURVES_EXP3; the legend names the algebra
def curves_exp2(method, tex):
    return [
        (((f"{method}, TT", "adaptive"),), rf"TT over ${tex}$", "x", 0, "-"),
        (((f"{method}, HOSVD", "adaptive"),), rf"HOSVD over ${tex}$", "s", 1, "-"),
        (((f"{method}, ST-HOSVD", "adaptive"),), rf"ST-HOSVD over ${tex}$", "o", 2, "-"),
    ] + CURVES_EXP3[3:6]


CURVES_EXP2_LON = curves_exp2("R[Z_64] on lon", r"\mathbb{R}[\mathbb{Z}_{64}]")                 # exp2_lon_format.csv
CURVES_EXP2_LONLAT = curves_exp2("R[Z_64] (x) DCT on lon, lat",                                 # exp2_lonlat_format.csv
                                 r"\mathbb{R}[\mathbb{Z}_{64}] \otimes \mathrm{DCT}_{32}")

# ERA5, tree tensor networks (exp2_lon_ttn.csv, exp2_era5.py --ttn), with TT and ST-HOSVD in the styles
# of CURVES_EXP2_LON (no HOSVD); TTN over R takes the color of TTN over K, dashed
_LON, _ZTEX = "R[Z_64] on lon", r"\mathbb{R}[\mathbb{Z}_{64}]"
_NO_HOSVD = [c for c in CURVES_EXP2_LON if not c[1].startswith("HOSVD")]
CURVES_EXP2_TTN = (
    [(((f"{_LON}, TTN column-EOF", "adaptive"),), rf"TTN over ${_ZTEX}$", "D", 3, "-")]
    + _NO_HOSVD[:2]
    + [(((f"{RR}, TTN column-EOF", "uniform"),), r"TTN over $\mathbb{R}$", "D", 3, "--")]
    + _NO_HOSVD[2:]
)


def curves_for(name):
    if name.startswith("exp10"):
        return CURVES_EXP10
    if name.startswith("exp3"):
        return CURVES_EXP3
    if name.startswith("exp2_lonlat"):
        return CURVES_EXP2_LONLAT
    if name.startswith("exp2_lon_ttn"):
        return CURVES_EXP2_TTN
    if name.startswith("exp2_lon_"):
        return CURVES_EXP2_LON
    raise ValueError(f"no curves for results/{name}.csv")


def load(path):
    pts = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            pts.setdefault((r["method"], r["truncation"]), []).append(
                (float(r["rel_error"]), float(r["compression_ratio"])))
    return {k: np.array(sorted(v)) for k, v in pts.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="results/<name>.csv")
    ap.add_argument("--xmax", type=float, default=0.1)
    ap.add_argument("--legend", default="upper left", help="legend position (matplotlib loc)")
    ap.add_argument("--xstep", type=float, default=None, help="spacing of the x-axis ticks (default: automatic)")
    ap.add_argument("--mark-gap", type=float, default=None,
                    help="smallest distance between drawn markers, as a fraction of the axes "
                         "(default 0.015, 0.08 for exp2 and exp3: six or seven curves of about 55 points)")
    ap.add_argument("--logx", action="store_true", help="log error axis (every point drawn)")
    ap.add_argument("--logy", action="store_true", help="log compression-ratio axis")
    ap.add_argument("--ymax", type=float, default=None, help="top of the compression-ratio axis (default: automatic)")
    ap.add_argument("--ncol", type=int, default=1, help="legend columns")
    ap.add_argument("--drop", default=None,
                    help="leave out these curves, comma-separated legend labels without TeX, e.g. 'TT over K'")
    ap.add_argument("--out", default=None, help="output path without extension (default figures/<name>_paper)")
    args = ap.parse_args()
    if args.mark_gap is None:
        args.mark_gap = 0.08 if args.name.startswith(("exp2", "exp3")) else 0.015

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
    })

    data = load(ROOT / "results" / f"{args.name}.csv")
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ymax, drawn = 0.0, []
    curves = [(k, lab, m, PALETTE[i], ls) for k, lab, m, i, ls in curves_for(args.name)]
    if args.drop:                              # the label with the $ signs removed, e.g. "TT over K"
        drop = set(args.drop.split(","))
        curves = [c for c in curves if c[1].replace("$", "") not in drop]
    for keys, label, marker, color, ls in curves:
        key = next((k for k in keys if k in data), None)
        if key is None:
            continue
        c = data[key]
        c = c[c[:, 1] >= 1]
        tiny = c[:, 0] < args.xmax / 400
        if tiny.any() and not args.logx:
            c = np.vstack([c[tiny][np.argmax(c[tiny, 1])], c[~tiny]])
        inside = c[:, 0] <= args.xmax
        ymax = max(ymax, c[inside, 1].max())
        if c[:, 0].max() > args.xmax:          # the curve leaves through the right edge
            ymax = max(ymax, np.interp(args.xmax, c[:, 0], c[:, 1]))
        drawn.append((c, ax.plot(c[:, 0], c[:, 1], color=color, ls=ls, marker=marker, lw=1.0, ms=5 if marker == "s" else 6,
                                 mfc="none" if marker != "x" else color, mew=1.0, label=label)[0]))
    step = 50 if ymax > 300 else 25 if ymax > 120 else 10
    ytop = args.ymax or step * np.ceil(1.05 * ymax / step)
    xmin = 0.8 * min(c[:, 0].min() for c, _ in drawn)
    xpos = (lambda x: np.log(x / xmin) / np.log(args.xmax / xmin)) if args.logx else (lambda x: x / args.xmax)
    for c, line in drawn:                      # thin the markers where the points crowd
        keep, last = [], None
        for t, (x, y) in enumerate(c):
            q = np.array([xpos(x), np.log(y) / np.log(ytop) if args.logy else y / ytop])
            if last is None or np.hypot(*(q - last)) >= args.mark_gap:
                keep.append(t)
                last = q
        line.set_markevery(keep)
    if args.logx:
        ax.set_xscale("log")
        ax.set_xlim(xmin, args.xmax)
    else:
        ax.set_xlim(0, args.xmax)
    if args.logy:
        ax.set_yscale("log")
        ax.set_ylim(1, ytop)
    else:
        ax.set_ylim(0, ytop)
    if not args.logx and (args.xmax >= 0.1 or args.xstep):
        step = args.xstep or (0.01 if args.xmax <= 0.1 else 0.025)
        ticks = np.arange(0, args.xmax + 1e-9 * args.xmax, step).round(6)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel(r"\textbf{Relative Error}")
    ax.set_ylabel(r"\textbf{Compression Ratio}")
    lines = [line for _, line in drawn]
    if args.ncol == 2 and args.name.startswith("exp3"):   # the curves over K first: matplotlib fills the left
        left = [h for h in lines if h.get_label().endswith("over $K$")]   # column first, so they head it, and
        lines = left + [h for h in lines if h not in left]                  # the columns have equal length
    ax.legend(handles=lines, ncol=args.ncol, fontsize=8, frameon=True, fancybox=False, loc=args.legend,
              edgecolor="black", framealpha=1, borderpad=0.3, handlelength=1.8, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    out = Path(args.out) if args.out else ROOT / "figures" / f"{args.name}_paper"
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300,     # log axes: keep the last tick label inside
                    **({"bbox_inches": "tight", "pad_inches": 0.02} if args.logx else {}))
    print(f"saved {out}.pdf and .png")


if __name__ == "__main__":
    main()
