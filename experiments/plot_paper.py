"""Paper figures: compression ratio against relative error, in the style of Kilmer, Horesh, Avron and
Newman (PNAS 2021), Fig. 3.  Each figure is drawn from results/<name>.csv into figures/<name>.pdf
and .png, with the curves and axes of FIGURES:

    fig5_quantum_circuits   Section 6.1, SVD over K = M_8(C)
    fig6_vlasov_poisson     Section 6.2, Tucker and TT over K = R[Z_32 x Z_32]
    fig8_era5_K1            Section 6.3, ST-HOSVD, TT and TTN over K_1 = R[Z_64]
    fig9_era5_K2            Section 6.3, HOSVD, ST-HOSVD and TT over K_2 = R[Z_64] (x) DCT_32

On a linear error axis, points with relative error below xmax / 400 are indistinguishable from 0;
of those only the one with the largest compression ratio is drawn (the rest would stack on the
y-axis).  Markers closer than mark_gap of the axes to the previous marker are skipped; the lines
still pass through every point.

Usage:
    python experiments/plot_paper.py                          # all four figures
    python experiments/plot_paper.py fig6_vlasov_poisson      # one figure
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# colorblind-safe palette (Paul Tol, bright), indexed by the color index of each curve
PALETTE = ["#EE6677", "#4477AA", "#228833", "#AA3377", "#000000", "#EE7733", "#33BBEE"]

# every curve: ((method, truncation) in the csv, legend label, marker, color index, line style)
# Figure 5: one color per method
CURVES_FIG5 = [
    (("SVD over K", "adaptive"), r"SVD over $K$", "x", 0, "-"),
    (("SVD over C of A_(1)", "uniform"), r"SVD over $\mathbb{C}$ of $A_{(1)}$", ">", 2, "-"),
    (("SVD over C of A_(2)", "uniform"), r"SVD over $\mathbb{C}$ of $A_{(2)}$", "o", 3, "-"),
    (("HOSVD over C of A", "uniform"), r"HOSVD over $\mathbb{C}$ of $A$", "s", 4, "-"),
]

# Figures 6, 8 and 9: one marker per format, solid over K, dashed over R, and a color of its own for
# every curve; the legend names the algebra (K in Figure 6)
RR = "R (no algebra)"
OVER_R = [
    ((f"{RR}, TT", "uniform"), r"TT over $\mathbb{R}$", "+", 6, "--"),
    ((f"{RR}, HOSVD", "uniform"), r"HOSVD over $\mathbb{R}$", "s", 4, "--"),
    ((f"{RR}, ST-HOSVD", "uniform"), r"ST-HOSVD over $\mathbb{R}$", "o", 5, "--"),
]


def over_k(method, tex):
    return [
        ((f"{method}, TT", "adaptive"), rf"TT over ${tex}$", "x", 0, "-"),
        ((f"{method}, HOSVD", "adaptive"), rf"HOSVD over ${tex}$", "s", 1, "-"),
        ((f"{method}, ST-HOSVD", "adaptive"), rf"ST-HOSVD over ${tex}$", "o", 2, "-"),
    ]


# Figure 6 leaves out TT over K (the text quotes it from the csv)
CURVES_FIG6 = over_k("R[Z_32 x Z_32] on x and y", "K")[1:] + OVER_R + [
    (("matrix SVD (x, y) | (vx, vy, t)", "rank"), r"SVD over $\mathbb{R}$ of $F_{(x,y)}$", "<", 3, "-."),
]
# Figure 8: no HOSVD; TTN over R takes the color of TTN over K_1, dashed
_K1, _K1TEX = "R[Z_64] on lon", r"\mathbb{R}[\mathbb{Z}_{64}]"
_TT1, _, _ST1 = over_k(_K1, _K1TEX)
CURVES_FIG8 = [
    ((f"{_K1}, TTN column-EOF", "adaptive"), rf"TTN over ${_K1TEX}$", "D", 3, "-"),
    _TT1, _ST1,
    ((f"{RR}, TTN column-EOF", "uniform"), r"TTN over $\mathbb{R}$", "D", 3, "--"),
    OVER_R[0], OVER_R[2],
]
CURVES_FIG9 = over_k("R[Z_64] (x) DCT on lon, lat", r"\mathbb{R}[\mathbb{Z}_{64}] \otimes \mathrm{DCT}_{32}") + OVER_R

# name: curves and axes.  xmax: right end of the error axis; xstep: its tick spacing (linear axis);
# ymax: top of the compression-ratio axis (default: automatic); logx, logy: log axes
FIGURES = {
    "fig5_quantum_circuits": dict(curves=CURVES_FIG5, xmax=0.1, logx=True, logy=True, ymax=2000, mark_gap=0.04),
    "fig6_vlasov_poisson": dict(curves=CURVES_FIG6, xmax=0.5, xstep=0.1, logy=True, legend="lower right", ncol=2),
    "fig8_era5_K1": dict(curves=CURVES_FIG8, xmax=0.5, xstep=0.1, logy=True),
    "fig9_era5_K2": dict(curves=CURVES_FIG9, xmax=0.5, xstep=0.1, logy=True, ymax=100),
}
DEFAULTS = dict(xstep=None, logx=False, logy=False, ymax=None, mark_gap=0.08, legend="upper left", ncol=1)


def load(path):
    pts = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            pts.setdefault((r["method"], r["truncation"]), []).append(
                (float(r["rel_error"]), float(r["compression_ratio"])))
    return {k: np.array(sorted(v)) for k, v in pts.items()}


def plot(name, outdir):
    import matplotlib.pyplot as plt

    f = {**DEFAULTS, **FIGURES[name]}
    xmax = f["xmax"]
    data = load(ROOT / "results" / f"{name}.csv")
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ymax, drawn = 0.0, []
    for key, label, marker, ci, ls in f["curves"]:
        c = data[key]
        c = c[c[:, 1] >= 1]
        tiny = c[:, 0] < xmax / 400
        if tiny.any() and not f["logx"]:
            c = np.vstack([c[tiny][np.argmax(c[tiny, 1])], c[~tiny]])
        inside = c[:, 0] <= xmax
        ymax = max(ymax, c[inside, 1].max())
        if c[:, 0].max() > xmax:               # the curve leaves through the right edge
            ymax = max(ymax, np.interp(xmax, c[:, 0], c[:, 1]))
        drawn.append((c, ax.plot(c[:, 0], c[:, 1], color=PALETTE[ci], ls=ls, marker=marker, lw=1.0,
                                 ms=5 if marker == "s" else 6, mfc="none" if marker != "x" else PALETTE[ci],
                                 mew=1.0, label=label)[0]))
    step = 50 if ymax > 300 else 25 if ymax > 120 else 10
    ytop = f["ymax"] or step * np.ceil(1.05 * ymax / step)
    xmin = 0.8 * min(c[:, 0].min() for c, _ in drawn)
    xpos = (lambda x: np.log(x / xmin) / np.log(xmax / xmin)) if f["logx"] else (lambda x: x / xmax)
    for c, line in drawn:                      # thin the markers where the points crowd
        keep, last = [], None
        for t, (x, y) in enumerate(c):
            q = np.array([xpos(x), np.log(y) / np.log(ytop) if f["logy"] else y / ytop])
            if last is None or np.hypot(*(q - last)) >= f["mark_gap"]:
                keep.append(t)
                last = q
        line.set_markevery(keep)
    if f["logx"]:
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(0, xmax)
    if f["logy"]:
        ax.set_yscale("log")
        ax.set_ylim(1, ytop)
    else:
        ax.set_ylim(0, ytop)
    if not f["logx"]:
        ticks = np.arange(0, xmax + 1e-9 * xmax, f["xstep"]).round(6)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel(r"\textbf{Relative Error}")
    ax.set_ylabel(r"\textbf{Compression Ratio}")
    lines = [line for _, line in drawn]
    if f["ncol"] == 2:                         # the curves over K first: matplotlib fills the left
        left = [h for h in lines if h.get_label().endswith("over $K$")]   # column first, so they head it, and
        lines = left + [h for h in lines if h not in left]                  # the columns have equal length
    ax.legend(handles=lines, ncol=f["ncol"], fontsize=8, frameon=True, fancybox=False, loc=f["legend"],
              edgecolor="black", framealpha=1, borderpad=0.3, handlelength=1.8, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    out = Path(outdir) / name
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300,     # log axes: keep the last tick label inside
                    **({"bbox_inches": "tight", "pad_inches": 0.02} if f["logx"] else {}))
    plt.close(fig)
    print(f"saved {out}.pdf and .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", metavar="name",
                    help=f"figures to draw, from {', '.join(FIGURES)} (default: all)")
    ap.add_argument("--outdir", default=str(ROOT / "figures"))
    args = ap.parse_args()
    unknown = [n for n in args.names if n not in FIGURES]
    if unknown:
        ap.error(f"unknown figure {', '.join(unknown)}; choose from {', '.join(FIGURES)}")

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
    for name in args.names or FIGURES:
        plot(name, args.outdir)


if __name__ == "__main__":
    main()
