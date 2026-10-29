"""Unitaries of random three-qubit circuits split into two halves (experiment 10).

Circuits.  Three qubits, so every circuit acts by a unitary in M_8(C).  A half-circuit is a
brickwork of four two-qubit gates, on qubits (0, 1), (1, 2), (0, 1), (1, 2), each gate drawn
from the Haar measure on U(4).  We draw two catalogues: m "second halves" and n "first
halves".  Running first half j and then second half i gives the unitary

    U(i, j) = U_second(i) U_first(j)        (the later gates act from the left).

Coherent noise with a common drift.  Each gate slot s carries a coherent error
exp(-i xi c_s H_s), with a fixed random Hermitian direction H_s (operator norm 1), a fixed
per-slot strength c_s (normal with standard deviation --eps), and one drift variable xi,
shared by all gates of both halves (a slowly varying calibration offset), with xi standard
normal.  For a fixed xi the circuit is again a product of the two halves, and the datum is
the expectation over the drift, computed exactly by Gauss-Hermite quadrature with --nodes
nodes xi_q and weights w_q (sum w_q = 1):

    A(i, j) = sum_q w_q U_second(i, xi_q) U_first(j, xi_q)  in  K = M_8(C),

optionally followed by complex Gaussian noise of relative size --noise (estimation noise;
0 by default, as in the paper).  For one value of xi, A has rank 1 over K (one column over K
times one row over K); the drift values are close to one another, so A is approximately
low-rank over K.  The drift must be shared:
with independent errors in the two halves, the expectation would factor as
E[U_second(i)] E[U_first(j)] and have rank exactly 1 over K.

The gates are generic (Haar), not functions of a few angles, so the 64 coordinates of the
half-circuit unitaries vary independently from one catalogue entry to the next.

Output: data/processed/circuits_noise0.npz with
    A  (m, n, 64) complex64, entries of the 8 x 8 matrices in row-major order (the
       coefficients of staralg.MatrixAlgebra(8, "C"))                  (128 MB for 500 x 500)
    nodes, weights (quadrature), strengths c_s (8,), seed, eps, noise
Usage:
    python data/simulate_circuits.py                  # 500 x 500, about a minute
    python data/simulate_circuits.py --m 300 --n 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NQ = 3
D = 2 ** NQ
SLOTS = (0, 1, 0, 1)          # first qubit of each two-qubit gate (brickwork on 3 qubits)


def haar_unitary(rng, d, size):
    """size Haar-random d x d unitaries: (size, d, d)."""
    Z = (rng.standard_normal((size, d, d)) + 1j * rng.standard_normal((size, d, d))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    ph = np.diagonal(R, axis1=-2, axis2=-1)
    return Q * (ph / np.abs(ph))[:, None, :]


def random_hermitian(rng, d):
    Z = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = (Z + Z.conj().T) / 2
    return H / np.linalg.norm(H, 2)


def expm_herm(H, a):
    """exp(-i a H) for Hermitian H."""
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * a * w)) @ V.conj().T


def embed(G, q):
    """A two-qubit gate (..., 4, 4) acting on qubits (q, q + 1) of three: (..., 8, 8)."""
    left, right = np.eye(2 ** q), np.eye(2 ** (NQ - q - 2))
    return np.einsum("ab,...cd,ef->...acebdf", left, G, right).reshape(G.shape[:-2] + (D, D))


def half_circuits(rng, size, H, amps):
    """Unitaries of `size` random half-circuits for every drift value: (nodes, size, 8, 8)."""
    gates = [haar_unitary(rng, 4, size) for _ in SLOTS]
    out = np.empty((amps.shape[0], size, D, D), dtype=complex)
    for x in range(amps.shape[0]):
        U = np.broadcast_to(np.eye(D), (size, D, D))
        for s, q in enumerate(SLOTS):
            G = gates[s] @ expm_herm(H[s], amps[x, s])        # gate followed by its coherent error
            U = embed(G, q) @ U                                 # later gates act from the left
        out[x] = U
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=500, help="number of second halves (rows)")
    ap.add_argument("--n", type=int, default=500, help="number of first halves (columns)")
    ap.add_argument("--nodes", type=int, default=20, help="Gauss-Hermite nodes for the expectation over the drift")
    ap.add_argument("--eps", type=float, default=0.08, help="standard deviation of the per-gate error strengths")
    ap.add_argument("--noise", type=float, default=0.0, help="relative estimation noise (0 in the paper)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "processed" / "circuits_noise0.npz"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    H_first = [random_hermitian(rng, 4) for _ in SLOTS]
    H_second = [random_hermitian(rng, 4) for _ in SLOTS]
    c = args.eps * rng.standard_normal(2 * len(SLOTS))                    # first half, then second half
    xi, w = np.polynomial.hermite_e.hermegauss(args.nodes)                # weight exp(-xi^2 / 2)
    w = w / w.sum()                                                       # expectation over xi ~ N(0, 1)
    amps = np.outer(xi, c)                                                # (nodes, 8): the drift scales every gate
    first = half_circuits(rng, args.n, H_first, amps[:, :len(SLOTS)])
    second = half_circuits(rng, args.m, H_second, amps[:, len(SLOTS):])

    # check: every half-circuit is unitary
    I = np.eye(D)
    err = max(np.abs(first[0] @ np.conj(np.swapaxes(first[0], -1, -2)) - I).max(),
              np.abs(second[0] @ np.conj(np.swapaxes(second[0], -1, -2)) - I).max())
    print(f"half-circuits: max |U U^* - I| = {err:.1e}")

    A = np.einsum("q,qiab,qjbc->ijac", w, second, first)                   # (m, n, 8, 8)
    noise = (rng.standard_normal(A.shape) + 1j * rng.standard_normal(A.shape)) / np.sqrt(2)
    A += args.noise * np.linalg.norm(A) / np.sqrt(A.size) * noise
    A = A.reshape(args.m, args.n, D * D).astype(np.complex64)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, A=A, nodes=xi, weights=w, strengths=c, seed=args.seed, eps=args.eps, noise=args.noise)
    print(f"saved {out}: A {A.shape} complex64 ({A.nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
