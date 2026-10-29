"""Quantum-circuit data: SVD of a matrix over K = M_8(C) against SVDs of its unfoldings and
against HOSVD over C (paper Section 6.1, Figure 5).

Data (data/simulate_circuits.py): A(i, j) = expectation over a common drift xi of
U_second(i, xi) U_first(j, xi), the unitaries of random three-qubit circuits split into two
halves, with coherent gate errors scaled by xi (Gauss-Hermite quadrature over xi ~ N(0, 1)).
A is an m x n matrix over the simple, noncommutative algebra K = M_8(C).  For one value of
xi, A has rank 1 over K; the expectation makes it approximately low-rank over K.

Methods:
    SVD over K                    uniform K-rank and adaptive truncation
    SVD over C of A_(1)           mode-1 unfolding of the m x n x 64 tensor, m x 64n
    SVD over C of A_(2)           mode-2 unfolding (computed on its transpose, 64m x n)
                                  (unfoldings: every rank up to 256, since their rank is about
                                  64 K-ranks and the default grid is geometric above 64)
    HOSVD over C of A             A as an m x n x 64 complex tensor, error-tolerance grid of 200
                                  tolerances from --eps-min to 0.5 (one row per distinct core)

Why the SVD over K should win here: the half-circuits are generic unitaries, so the 64
coordinates of U_second(i) vary independently with i.  Tucker then finds no small scalar
subspace shared by the coordinates and needs about 64 dimensions per side and a dense
64 x 64 x 64 core, while one column over K stores each half once.

Output: results/fig5_quantum_circuits.csv
Usage:
    python data/simulate_circuits.py                 # data/processed/circuits.npz
    python experiments/quantum_circuits.py           # a few minutes
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from staralg import MatrixAlgebra, Trivial, compression_ratio, hosvd_K, svd_K

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["method", "truncation", "param", "rep", "n_params", "compression_ratio", "rel_error"]
TARGET_RATIOS = [2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]


def rel(X, Y):
    return float(np.linalg.norm((X - Y).ravel()) / np.linalg.norm(X.ravel()))


def run(label, A, K, n_data, ref=None, back=None, adaptive=True, kmax=200, extra_ranks=()):
    """SVD over K of A: every rank up to 64 and a geometric grid beyond (uniform), and, if adaptive,
    per-block truncations within storage budgets.  If A is a rearrangement of the data, back maps a
    reconstruction to the shape of ref, the data the errors are measured against."""
    svd = svd_K(A, K)
    ref = A if ref is None else ref
    back = (lambda Y: Y) if back is None else back
    rows = []
    top = min(kmax, svd.m, svd.n)
    # every rank up to 64 (the linear-axis figure needs the small ranks), geometric beyond
    ks = sorted(set(np.unique(np.round(np.geomspace(1, top, 30)).astype(int))) | set(range(1, min(64, top) + 1))
                | {int(k) for k in extra_ranks if 1 <= k <= top})   # extra_ranks: e.g. a denser grid above 64
    for k in ks:
        p = svd.n_params(rank=k)
        rows.append([label, "uniform", int(k), 0, p, compression_ratio(n_data, p), rel(ref, back(svd.reconstruct(rank=k)))])
    if adaptive:
        seen = set()
        for ratio in sorted(set(TARGET_RATIOS) | set(np.round(np.geomspace(2, 4096, 160), 2))):
            budget = n_data / ratio
            p = svd.n_params(budget=budget)
            if p == 0 or p in seen:
                continue
            seen.add(p)
            rows.append([label, "adaptive", ratio, 0, p, compression_ratio(n_data, p),
                         rel(ref, back(svd.reconstruct(budget=budget)))])
    print(f"  {label}: done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "circuits.npz"))
    ap.add_argument("--out", default=str(ROOT / "results" / "fig5_quantum_circuits.csv"))
    ap.add_argument("--eps-min", type=float, default=1e-8,
                    help="smallest HOSVD tolerance (below the float32 accuracy of the data; "
                         "raise it to the noise level for noisy data)")
    args = ap.parse_args()

    A = np.load(args.data)["A"].astype(np.complex128)              # (m, n, 64)
    m, n, dk = A.shape
    D = int(round(np.sqrt(dk)))
    n_data = 2 * A.size
    print(f"matrix over K = M_{D}(C) of size {m} x {n}")
    K = MatrixAlgebra(D, "C")
    C = Trivial("C")
    kmax = min(m, n, 400)

    rows = []
    rows += run("SVD over K", A, K, n_data)
    dense = range(65, 257)
    rows += run("SVD over C of A_(1)", A.reshape(m, n * dk), C, n_data,
                ref=A, back=lambda Y: Y.reshape(m, n, dk), adaptive=False, kmax=kmax, extra_ranks=dense)
    rows += run("SVD over C of A_(2)", A.transpose(0, 2, 1).reshape(m * dk, n), C, n_data,
                ref=A, back=lambda Y: Y.reshape(m, dk, n).transpose(0, 2, 1), adaptive=False, kmax=kmax,
                extra_ranks=dense)

    H = hosvd_K(A, C)
    print("  HOSVD over C of A: core sizes along the tolerance grid (first tolerance giving each core)")
    seen = set()
    for e in np.geomspace(args.eps_min, 0.5, 200):
        T = H.truncate(eps=float(e))
        if T.ranks_K in seen:                                       # same core as a smaller tolerance
            continue
        seen.add(T.ranks_K)
        p = T.n_params()
        err = rel(A, T.reconstruct())
        rows.append(["HOSVD over C of A", "uniform", float(e), 0, p, compression_ratio(n_data, p), err])
        print(f"    eps {e:.5f}: core {T.ranks_K}, compression ratio {n_data / p:7.1f}, error {err:.5f}")
    print("  HOSVD over C of A: done", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")

    from common import summarize
    srows = [[r[0], "Equal storage", "", r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows]
    summarize(srows, targets=(8, 16, 32, 64, 128, 200))
    print("\nEQUAL RANK: relative error at K-rank k (over K) or rank k (unfoldings)")
    for k in (1, 2, 4, 8, 16):
        vals = {r[0]: r[6] for r in rows if r[1] == "uniform" and r[2] == k and not r[0].startswith("HOSVD")}
        print(f"  k={k:2d}: " + "   ".join(f"{name}: {v:.5f}" for name, v in vals.items()))


if __name__ == "__main__":
    main()
