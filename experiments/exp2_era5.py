"""ERA5 anomalies: Tucker, TT and tree tensor networks over R[Z_64] and R[Z_64] (x) DCT_32
(paper Figures 7 and 8, and the tree tensor network figure).

Data (data/fetch_era5.py): X with modes lon (64), lat (32), level (13), var (5),
time (365), standardized daily anomalies for 2019.

Configurations (which modes go into the algebra; the rest are modes of the tensor over K):
    none       K = R                         lon x lat x level x var x t   (order 5)
    lon        K = R[Z_64], DFT t-product    lat x level x var x t         (order 4)
    lon + lat  K = R[Z_64] (x) DCT_32        level x var x t               (order 3)

--format: two experiments, each against the same real baseline, on the fine tolerance grid
FORMAT_EPS (0.005 to 0.6; HOSVD to 0.8):
    results/exp2_lon_format.csv      TT, HOSVD, ST-HOSVD over K = R[Z_64] on lon (adaptive)
                                     and over R (no algebra)
    results/exp2_lonlat_format.csv   TT, HOSVD, ST-HOSVD over K = R[Z_64] (x) DCT_32 on lon, lat
                                     (adaptive) and over R (the same rows as in the first file)
TT takes the modes in order (lat, level, var, time over R[Z_64]; level, var, time over
R[Z_64] (x) DCT_32; lon, lat, level, var, time over R).

--ttn: tree tensor networks (TTN-SVD, Algorithm 6) on the same fine grid, written to
results/exp2_lon_ttn.csv, with the TT, HOSVD and ST-HOSVD rows of results/exp2_lon_format.csv copied in
(run --format first).  The column-EOF tree over R[Z_64] on lon (modes lat, level, var, time),
    (((lat), (level, var)), time): level and variable meet at a "column" vertex (the vertical
    profile of the state), latitude joins it at a "space" vertex, and the root splits space
    from time, as in an EOF analysis.  It gives the same approximation as Tucker of the
    reshaped lat x (level, var) x time tensor.
and the same tree over R with a lon leaf at the space vertex, (((lon), (lat), (level, var)), time).
Adaptive per-block ranks over R[Z_64], uniform over R.

Usage:
    python data/fetch_era5.py
    python experiments/exp2_era5.py               # --format, then --ttn (about 40 minutes on a laptop)
    python experiments/exp2_era5.py --format      # the two format comparisons (about 30 minutes)
    python experiments/exp2_era5.py --ttn         # the tree tensor networks (about 7 minutes)
    python experiments/exp2_era5.py --quick       # every fourth tolerance, for a first look
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import FORMAT_EPS, FORMAT_EPS_HOSVD, Recorder, load_rows, summarize, write_csv
from staralg import TensorProduct, Trivial, TubeDCT, TubeDFT, hosvd_K, sthosvd_K, tt_svd_K, ttn_svd_K

ROOT = Path(__file__).resolve().parents[1]

# --ttn: trees over R[Z_64] on lon (modes lat 0, level 1, var 2, time 3), and over R (lon 0, lat 1,
# level 2, var 3, time 4); see the module docstring
TREES_EXP2 = [
    ("column-EOF", (((0,), (1, 2)), 3)),
]
TREE_EXP2_R = ("column-EOF", (((0,), (1,), (2, 3)), 4))


def configurations(nlon, nlat):
    """(name, K, alg_axes, kind) of the real baseline, lon, and lon + lat."""
    return [
        ("R (no algebra)", Trivial("R"), None, "baseline"),
        (f"R[Z_{nlon}] on lon", TubeDFT(nlon), 0, "ours"),
        (f"R[Z_{nlon}] (x) DCT on lon, lat", TensorProduct(TubeDFT(nlon), TubeDCT(nlat)), (0, 1), "ours"),
    ]


def format_sweep(X, outs, eps_grid=FORMAT_EPS):
    """--format: TT, HOSVD and ST-HOSVD over R[Z_64] on lon and over R[Z_64] (x) DCT_32 on lon, lat
    (adaptive), and over R, on a fine tolerance grid.  outs: the csv paths of the two experiments,
    rewritten after every method (the R rows go into both)."""
    nlon, nlat = X.shape[:2]
    rec, panel = Recorder(X.size), "Format comparison"
    real_cfg, lon_cfg, lonlat_cfg = configurations(nlon, nlat)
    lon, lonlat, real = lon_cfg[0], lonlat_cfg[0], real_cfg[0]

    def write():
        for out, name in zip(outs, (lon, lonlat)):
            write_csv([r for r in rec.rows if r[0].startswith((name + ",", real + ","))], out)
    nX = np.linalg.norm(X)

    def add(method, kind, trunc, e, T):
        err = np.linalg.norm(X - T.reconstruct()) / nX
        rec.add(method, panel, kind, trunc, e, T.n_params(), err)
        print(f"  {method:40s} eps {e:<7g} compression ratio {X.size / T.n_params():9.1f}, error {err:.4f}", flush=True)

    for name, K, ax, kind in (lon_cfg, lonlat_cfg, real_cfg):
        trunc = "uniform" if ax is None else "adaptive"     # over R (one block) the two coincide
        kw = {} if ax is None else dict(adaptive=True)
        t0 = time.time()
        h = hosvd_K(X, K, alg_axes=ax)
        for e in eps_grid + FORMAT_EPS_HOSVD:
            add(f"{name}, HOSVD", kind, trunc, e, h.truncate(eps=e, **kw))
        del h
        write()
        print(f"  {name}, HOSVD done ({time.time() - t0:.0f} s)", flush=True)
        for label, run in (("ST-HOSVD", lambda e: sthosvd_K(X, K, eps=e, alg_axes=ax, **kw)),
                           ("TT", lambda e: tt_svd_K(X, K, eps=e, alg_axes=ax, **kw))):
            t0 = time.time()
            for e in eps_grid:
                add(f"{name}, {label}", kind, trunc, e, run(e))
            write()
            print(f"  {name}, {label} done ({time.time() - t0:.0f} s)", flush=True)
    summarize(rec.rows)


def ttn_sweep(X, out, eps_grid=FORMAT_EPS):
    """--ttn: TTN-SVD on the trees of TREES_EXP2 over R[Z_64] on lon (adaptive) and on TREE_EXP2_R over
    R, on a fine tolerance grid; the TT, HOSVD and ST-HOSVD rows of exp2_lon_format.csv are copied in.
    out is rewritten after every tree."""
    nlon, nlat = X.shape[:2]
    rec, panel = Recorder(X.size), "Tree tensor networks"
    (real, _, _, _), (lon, K, ax, kind), _ = configurations(nlon, nlat)
    fmt = Path(out).with_name("exp2_lon_format.csv")
    old = [[r[0], panel] + r[2:] for r in load_rows(fmt) if r[0].endswith((", TT", ", HOSVD", ", ST-HOSVD"))] \
        if fmt.exists() else []
    if not old:
        print(f"  no TT, HOSVD or ST-HOSVD rows in {fmt}: run --format first to compare with them")
    nX = np.linalg.norm(X)
    runs = [(f"{lon}, TTN {label}", K, ax, kind, tree, dict(adaptive=True), "adaptive") for label, tree in TREES_EXP2]
    runs.append((f"{real}, TTN {TREE_EXP2_R[0]}", Trivial("R"), None, "baseline", TREE_EXP2_R[1], {}, "uniform"))
    for method, K, ax, kind, tree, kw, trunc in runs:
        t0 = time.time()
        for e in eps_grid:
            T = ttn_svd_K(X, K, tree, eps=e, alg_axes=ax, **kw)
            err = np.linalg.norm(X - T.reconstruct()) / nX
            rec.add(method, panel, kind, trunc, e, T.n_params(), err)
            print(f"  {method:45s} eps {e:<7g} compression ratio {X.size / T.n_params():9.1f}, error {err:.4f}", flush=True)
        write_csv(old + rec.rows, out)
        print(f"  {method} done ({time.time() - t0:.0f} s)", flush=True)
    summarize(old + rec.rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "era5_2019.npz"))
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder of the csv files")
    ap.add_argument("--quick", action="store_true", help="every fourth tolerance of the grid")
    ap.add_argument("--format", action="store_true",
                    help="only the two format comparisons (results/exp2_lon_format.csv, results/exp2_lonlat_format.csv)")
    ap.add_argument("--ttn", action="store_true",
                    help="only the tree tensor networks (results/exp2_lon_ttn.csv)")
    args = ap.parse_args()
    if not (args.format or args.ttn):
        args.format = args.ttn = True

    X = np.load(args.data)["X"].astype(np.float64)
    print(f"loaded {args.data}: X {X.shape}")
    eps = FORMAT_EPS[::4] if args.quick else FORMAT_EPS
    res = Path(args.results)
    if args.format:
        format_sweep(X, [res / f"exp2_{k}_format.csv" for k in ("lon", "lonlat")], eps_grid=eps)
    if args.ttn:
        ttn_sweep(X, res / "exp2_lon_ttn.csv", eps_grid=eps)


if __name__ == "__main__":
    main()
