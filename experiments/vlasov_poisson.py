"""Vlasov-Poisson data: Tucker and TT over R[Z_32 x Z_32] (paper Section 6.2, Figure 6).

Data (data/simulate_vlasov.py): f(x, y, v_x, v_y, t), 32 x 32 x 48 x 48 x 32, nonlinear
Landau damping on a box that is periodic in x and in y.  The Vlasov-Poisson flow commutes
with translations in both directions, so the translation group of the box, Z_32 x Z_32,
is an exact symmetry, and the matching algebra is

    K = R[Z_32 x Z_32] = R[Z_32] (x) R[Z_32]     (structure map: the 2D DFT in x, y),

with one 1 x 1 block per spatial wavevector (m, n).  f is an order-3 tensor
v_x x v_y x t over K.

Methods, on the fine tolerance grid FORMAT_EPS (0.005 to 0.6; HOSVD to 0.8, as its error stays
well below the tolerance, so that every curve crosses relative error 0.5):
    TT, HOSVD, ST-HOSVD over K = R[Z_32 x Z_32] (adaptive per-block ranks)
    TT, HOSVD, ST-HOSVD over R (the real 32 x 32 x 48 x 48 x 32 tensor)
    matrix SVD of the unfolding (x, y) x (v_x, v_y, t), at the smallest rank meeting each tolerance
TT takes the modes in order: (v_x, v_y, t) over K, (x, y, v_x, v_y, t) over R.

Output: results/fig6_vlasov_poisson.csv
Usage:
    python data/simulate_vlasov.py
    python experiments/vlasov_poisson.py         # about 20 minutes on an 8 GB laptop
    python experiments/vlasov_poisson.py --quick  # every fourth tolerance, for a first look
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import FORMAT_EPS, FORMAT_EPS_HOSVD, Recorder, summarize, write_csv
from staralg import AbelianGroupAlgebra, Trivial, hosvd_K, sthosvd_K, svd_K, tt_svd_K

ROOT = Path(__file__).resolve().parents[1]

XY = "R[Z_32 x Z_32] on x and y"
RR = "R (no algebra)"


def format_sweep(X, Kxy, R, out, eps_grid=FORMAT_EPS):
    """TT, HOSVD and ST-HOSVD over K (adaptive) and over R, and the matrix SVD of the (x, y)
    unfolding, on a fine tolerance grid.  The csv is rewritten after every method."""
    rec, panel = Recorder(X.size), "Format comparison"
    nX = np.linalg.norm(X)

    def add(method, kind, trunc, e, T):
        err = np.linalg.norm(X - T.reconstruct()) / nX
        rec.add(method, panel, kind, trunc, e, T.n_params(), err)
        print(f"  {method:40s} eps {e:<7g} compression ratio {X.size / T.n_params():9.1f}, error {err:.4f}", flush=True)

    for name, KK, ax, kind, trunc in ((XY, Kxy, (0, 1), "ours", "adaptive"),
                                      (RR, R, None, "baseline", "uniform")):
        kw = dict(adaptive=True) if trunc == "adaptive" else {}
        t0 = time.time()
        h = hosvd_K(X, KK, alg_axes=ax)
        for e in eps_grid + FORMAT_EPS_HOSVD:
            add(f"{name}, HOSVD", kind, trunc, e, h.truncate(eps=e, **kw))
        del h
        write_csv(rec.rows, out)
        print(f"  {name}, HOSVD done ({time.time() - t0:.0f} s)", flush=True)
        for label, run in (("ST-HOSVD", lambda e: sthosvd_K(X, KK, eps=e, alg_axes=ax, **kw)),
                           ("TT", lambda e: tt_svd_K(X, KK, eps=e, alg_axes=ax, **kw))):
            t0 = time.time()
            for e in eps_grid:
                add(f"{name}, {label}", kind, trunc, e, run(e))
            write_csv(rec.rows, out)
            print(f"  {name}, {label} done ({time.time() - t0:.0f} s)", flush=True)

    # matrix SVD: for each tolerance the smallest rank with tail error <= eps ||X||
    A = X.reshape(X.shape[0] * X.shape[1], -1)
    svd = svd_K(A, R)
    s2 = np.sort(np.asarray(svd.S[0]).ravel().astype(np.float64))[::-1] ** 2
    tail = np.sqrt(np.append(np.cumsum(s2[::-1])[::-1], 0.0)) / np.sqrt(s2.sum())   # tail[r]: error at rank r
    ranks = sorted({int(np.argmax(tail <= e)) for e in eps_grid} - {0})
    for r in ranks:
        err = np.linalg.norm(X - svd.reconstruct(rank=r).reshape(X.shape)) / nX
        p = svd.n_params(rank=r)
        rec.add("matrix SVD (x, y) | (vx, vy, t)", panel, "baseline", "rank", r, p, err)
        print(f"  matrix SVD rank {r:4d}: compression ratio {X.size / p:9.1f}, error {err:.4f}", flush=True)
    write_csv(rec.rows, out)
    summarize(rec.rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "vlasov.npz"))
    ap.add_argument("--out", default=str(ROOT / "results" / "fig6_vlasov_poisson.csv"))
    ap.add_argument("--quick", action="store_true", help="every fourth tolerance of the grid")
    args = ap.parse_args()

    X = np.load(args.data)["f"].astype(np.float32)      # (x, y, vx, vy, t)
    nx, ny = X.shape[:2]
    print(f"loaded {args.data}: f {X.shape} ({X.nbytes / 1e6:.0f} MB as float32)")
    Kxy = AbelianGroupAlgebra(nx, ny)      # = TensorProduct(TubeDFT(nx), TubeDFT(ny))
    format_sweep(X, Kxy, Trivial("R"), args.out, eps_grid=FORMAT_EPS[::4] if args.quick else FORMAT_EPS)


if __name__ == "__main__":
    main()
