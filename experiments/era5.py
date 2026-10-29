"""ERA5 climate data (paper Section 6.3, Figures 8 and 9).

Data (data/fetch_era5.py): A with modes lon (64), lat (32), level (13), var (5),
time (365), standardized daily anomalies for 2019.

Algebras (which modes go into the algebra; the rest are modes of the tensor over K):
    none       K = R                         lon x lat x level x var x t   (order 5)
    lon        K_1 = R[Z_64], DFT t-product  lat x level x var x t         (order 4)
    lon + lat  K_2 = R[Z_64] (x) DCT_32      level x var x t               (order 3)

Methods, on the fine tolerance grid FORMAT_EPS (0.005 to 0.6; HOSVD to 0.8), with adaptive
per-block ranks over K_1 and K_2 and uniform ranks over R:
    results/fig8_era5_K1.csv   ST-HOSVD, TT-SVD and TTN-SVD over K_1, and over R
    results/fig9_era5_K2.csv   HOSVD, ST-HOSVD and TT-SVD over K_2, and over R
TT takes the modes in order (lat, level, var, time over K_1; level, var, time over K_2;
lon, lat, level, var, time over R).

TTN-SVD (Algorithm 6) uses the trees of paper Figure 7.  Over K_1 (modes lat, level, var, time),
    (((lat), (level, var)), time): level and variable meet at a "column" vertex (the vertical
    profile of the state), latitude joins it at a "space" vertex, and the root splits space
    from time, as in an EOF analysis.  It gives the same approximation as Tucker of the
    reshaped lat x (level, var) x time tensor.
Over R, the same tree with a lon leaf at the space vertex, (((lon), (lat), (level, var)), time).

Usage:
    python data/fetch_era5.py
    python experiments/era5.py               # about 35 minutes on a laptop
    python experiments/era5.py --quick       # every fourth tolerance, for a first look
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import FORMAT_EPS, FORMAT_EPS_HOSVD, Recorder, summarize, write_csv
from staralg import TensorProduct, Trivial, TubeDCT, TubeDFT, hosvd_K, sthosvd_K, tt_svd_K, ttn_svd_K

ROOT = Path(__file__).resolve().parents[1]

# paper Figure 7: the trees over K_1 (modes lat 0, level 1, var 2, time 3) and over R (lon 0, lat 1,
# level 2, var 3, time 4)
TREE_K1 = ("column-EOF", (((0,), (1, 2)), 3))
TREE_R = ("column-EOF", (((0,), (1,), (2, 3)), 4))


def sweep(X, out_k1, out_k2, eps_grid=FORMAT_EPS):
    """All decompositions of Figures 8 and 9 on a fine tolerance grid; both csv files are rewritten
    after every method (the rows over R go into both, each with the methods of its figure)."""
    nlon, nlat = X.shape[:2]
    rec, panel = Recorder(X.size), "Format comparison"
    K1 = (f"R[Z_{nlon}] on lon", TubeDFT(nlon), 0, "ours")
    K2 = (f"R[Z_{nlon}] (x) DCT on lon, lat", TensorProduct(TubeDFT(nlon), TubeDCT(nlat)), (0, 1), "ours")
    RR = ("R (no algebra)", Trivial("R"), None, "baseline")
    fig8 = {f"{n}, {m}" for n in (K1[0], RR[0]) for m in ("ST-HOSVD", "TT", f"TTN {TREE_K1[0]}")}
    fig9 = {f"{n}, {m}" for n in (K2[0], RR[0]) for m in ("HOSVD", "ST-HOSVD", "TT")}

    def write():
        write_csv([r for r in rec.rows if r[0] in fig8], out_k1)
        write_csv([r for r in rec.rows if r[0] in fig9], out_k2)
    nX = np.linalg.norm(X)

    def add(method, kind, trunc, e, T):
        err = np.linalg.norm(X - T.reconstruct()) / nX
        rec.add(method, panel, kind, trunc, e, T.n_params(), err)
        print(f"  {method:45s} eps {e:<7g} compression ratio {X.size / T.n_params():9.1f}, error {err:.4f}", flush=True)

    def run(method, kind, trunc, decompose, grid):
        t0 = time.time()
        for e in grid:
            add(method, kind, trunc, e, decompose(e))
        write()
        print(f"  {method} done ({time.time() - t0:.0f} s)", flush=True)

    # Tucker and TT: HOSVD over K_2 and over R only (Figure 9), ST-HOSVD and TT everywhere
    for name, K, ax, kind in (K1, K2, RR):
        trunc = "uniform" if ax is None else "adaptive"     # over R (one block) the two coincide
        kw = {} if ax is None else dict(adaptive=True)
        if name != K1[0]:
            h = hosvd_K(X, K, alg_axes=ax)
            run(f"{name}, HOSVD", kind, trunc, lambda e: h.truncate(eps=e, **kw), eps_grid + FORMAT_EPS_HOSVD)
            del h
        run(f"{name}, ST-HOSVD", kind, trunc, lambda e: sthosvd_K(X, K, eps=e, alg_axes=ax, **kw), eps_grid)
        run(f"{name}, TT", kind, trunc, lambda e: tt_svd_K(X, K, eps=e, alg_axes=ax, **kw), eps_grid)

    # tree tensor networks (Figure 8)
    for (name, K, ax, kind), (label, tree) in ((K1, TREE_K1), (RR, TREE_R)):
        trunc, kw = ("uniform", {}) if ax is None else ("adaptive", dict(adaptive=True))
        run(f"{name}, TTN {label}", kind, trunc, lambda e: ttn_svd_K(X, K, tree, eps=e, alg_axes=ax, **kw), eps_grid)
    summarize(rec.rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "era5_2019.npz"))
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder of the csv files")
    ap.add_argument("--quick", action="store_true", help="every fourth tolerance of the grid")
    args = ap.parse_args()

    X = np.load(args.data)["X"].astype(np.float64)
    print(f"loaded {args.data}: X {X.shape}")
    res = Path(args.results)
    sweep(X, res / "fig8_era5_K1.csv", res / "fig9_era5_K2.csv", eps_grid=FORMAT_EPS[::4] if args.quick else FORMAT_EPS)


if __name__ == "__main__":
    main()
