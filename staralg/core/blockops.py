"""Block-domain building blocks shared by the Tucker and tensor-train code.

* ``left_singular``: leading left singular vectors and all singular values of a batch
  of block matrices, quaternion-structured for 'H' blocks, using the Gram matrix
  M M^H when the matrices are wide (much cheaper in memory and time).
* ``select_counts``: how many singular values to keep in each block, for a uniform
  K-rank, a uniform rank chosen by a tolerance, or an adaptive (per-block) choice by
  a tolerance.  Counts are always multiples of the unit (2 for quaternion blocks,
  whose singular values come in equal pairs).
* ``mode_product``: batched mode-k products of block tensors.
"""
from __future__ import annotations

import numpy as np

from ..algebras.base import Group
from .svd import _block_svd, _structure_pairs

GRAM_RATIO = 2  # use the Gram matrix when cols > GRAM_RATIO * rows


def unit(g: Group) -> int:
    """Singular values that are kept or discarded together in a block of this group."""
    return 2 if g.division == "H" else 1


def left_singular(M: np.ndarray, g: Group):
    """U (B, rows, r) and singular values S (B, r), r = min(rows, cols), sorted decreasingly."""
    rows, cols = M.shape[-2:]
    if cols > GRAM_RATIO * rows:
        G = M @ np.conj(np.swapaxes(M, -1, -2))
        G = 0.5 * (G + np.conj(np.swapaxes(G, -1, -2)))
        dt = np.complex128 if np.iscomplexobj(G) else np.float64
        w, V = np.linalg.eigh(G.astype(dt))
        w, V = w[:, ::-1], V[:, :, ::-1]
        s = np.sqrt(np.clip(w, 0.0, None))
        if g.division == "H":
            V, s, _ = _structure_pairs(V, s)
        return V.astype(np.result_type(M.dtype, V.dtype), copy=False), s
    u, s, _ = _block_svd(M, g)
    return u, s


def tail_energy(spectra, groups, counts) -> float:
    """sum over blocks of c_rho * (sum of discarded sigma^2)."""
    tail = 0.0
    for S, g, c in zip(spectra, groups, counts):
        r = S.shape[-1]
        discard = np.arange(r)[None, :] >= c[:, None]
        tail += float(np.sum(g.weights[:, None] * S.astype(np.float64) ** 2 * discard))
    return tail


def uniform_counts(spectra, groups, rank: int):
    """Uniform K-rank: rank * s singular values in every block (capped by what exists)."""
    return [np.full(S.shape[0], min(int(rank) * g.m, S.shape[-1]), dtype=int) for S, g in zip(spectra, groups)]


def select_counts(spectra, groups, rank=None, delta2=None, adaptive=False, units=None):
    """Number of kept singular values per block, as a list of (B,) integer arrays.

    rank      uniform K-rank
    delta2    discarded weighted energy allowed: the smallest uniform K-rank with tail
              <= delta2 (adaptive=False), or per-block counts discarding the globally
              smallest units while the total stays <= delta2 (adaptive=True)
    units     optional list (one per group) of the number of singular values kept or
              discarded together in the adaptive rule (default: unit(g), i.e. quaternion
              pairs).  Tucker over a noncommutative K uses g.m, so that every block keeps
              whole K-columns.
    """
    unit_sizes = None if units is None else [int(u) for u in units]
    if rank is not None:
        return uniform_counts(spectra, groups, rank)
    if delta2 is None:
        return [np.full(S.shape[0], S.shape[-1], dtype=int) for S in spectra]
    if not adaptive:
        kmax = max(int(np.ceil(S.shape[-1] / g.m)) for S, g in zip(spectra, groups))
        for r in range(0, kmax + 1):
            c = uniform_counts(spectra, groups, r)
            if tail_energy(spectra, groups, c) <= delta2 * (1 + 1e-12):
                return c
        return uniform_counts(spectra, groups, kmax)
    units = []  # (energy, -t, group, block)
    counts = []
    for gi, (S, g) in enumerate(zip(spectra, groups)):
        u = unit(g) if unit_sizes is None else unit_sizes[gi]
        B, r = S.shape
        nu = r // u
        counts.append(np.full(B, nu, dtype=int))
        e = (S[:, : nu * u].astype(np.float64) ** 2).reshape(B, nu, u).sum(-1) * g.weights[:, None]
        for b in range(B):
            for t in range(nu):
                units.append((e[b, t], -t, gi, b))
    units.sort()
    acc = 0.0
    for e, _, gi, b in units:
        if acc + e > delta2 * (1 + 1e-12):
            break
        acc += e
        counts[gi][b] -= 1
    return [c * (unit(g) if unit_sizes is None else unit_sizes[gi]) for gi, (c, g) in enumerate(zip(counts, groups))]


def masked(U: np.ndarray, counts: np.ndarray, g: Group) -> np.ndarray:
    """First max(counts) columns of U, with the columns beyond each block's count zeroed.

    At least one unit of columns is kept so that shapes stay nonzero.
    """
    R = max(int(counts.max()), unit(g))
    R = min(R, U.shape[-1])
    keep = np.arange(R)[None, :] < counts[:, None]          # (B, R)
    return U[..., :R] * keep[:, None, :]


def mode_product(T: np.ndarray, M: np.ndarray, k: int) -> np.ndarray:
    """Batched mode-k product: T (B, n_0, ..., n_{N-1}), M (B, p, n_k) -> n_k replaced by p."""
    Tk = np.moveaxis(T, k + 1, -1)
    shp = Tk.shape
    out = Tk.reshape(shp[0], -1, shp[-1]) @ np.swapaxes(M, -1, -2)
    out = out.reshape(shp[:-1] + (M.shape[1],))
    return np.moveaxis(out, -1, k + 1)
