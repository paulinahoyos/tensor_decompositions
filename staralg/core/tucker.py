"""Tucker decompositions over a commutative K: HOSVD and ST-HOSVD.

    hosvd = hosvd_K(X, K, alg_axes=...)          # factor SVDs once
    T = hosvd.truncate(ranks=(4, 4, 2))          # uniform K-ranks
    T = hosvd.truncate(eps=0.1)                  # smallest uniform ranks with error <= eps ||X||
    T = hosvd.truncate(eps=0.1, adaptive=True)   # per-block ranks, error <= eps ||X||
    T = sthosvd_K(X, K, ranks=..., eps=..., adaptive=..., order=...)
    X_hat = T.reconstruct();  T.n_params();  T.tails

Why only commutative K.  For noncommutative K the mode products do not commute, the
quasi-optimality proof of the HOSVD fails, and the bound is false in general
(explicit quaternion counterexample in the paper; see also Ya and Yang,
arXiv:2309.05211).  A very proper commutative K is isomorphic to R^a (+) C^b, so all
blocks are 1 x 1 and a tensor over K is a stack of real and complex tensors, one per
block; the Tucker decomposition over K is a Tucker decomposition of every block with
shared (uniform) or separate (adaptive) ranks.

Truncation and error.  With tails tau_k (weighted discarded energy of mode k):
  HOSVD      tau_k <= ||X - X_hat||^2 <= sum_k tau_k
  ST-HOSVD   ||X - X_hat||^2 = sum_k tau_k (tau_k from the sequentially compressed core)
With a tolerance eps, every mode may discard eps^2 ||X||^2 / N, so ||X - X_hat|| <= eps ||X||.

Storage (real parameters): for each block, (core + factors) =
  params_per_entry * (prod_k R_k + sum_k n_k R_k);
for uniform K-ranks r this is (prod_k r_k + sum_k n_k r_k) dim_R K.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..algebras.base import Algebra, move_alg_axes_back, move_alg_axes_last
from .blockops import left_singular, masked, mode_product, select_counts, tail_energy


class NoncommutativeTuckerError(ValueError):
    """Raised when a Tucker decomposition is requested over a noncommutative K."""


def _check_commutative(K: Algebra, groups) -> None:
    if not K.is_commutative or any(g.m != 1 for g in groups):
        raise NoncommutativeTuckerError(
            f"Tucker decompositions over K are implemented by default only for commutative K; "
            f"{K.name} is noncommutative.  For noncommutative K the HOSVD is not "
            f"quasi-optimal in general (see the paper's counterexample).  Use tt_svd_K, "
            f"which is valid for every K, or pass allow_noncommutative=True for the "
            f"mirror-order Tucker of tucker_nc.py (ST-HOSVD keeps its error identity "
            f"||X - X_hat||^2 = sum of tails; HOSVD has no a priori bound)."
        )


def _prepare(X, K, alg_axes):
    X = np.asarray(X)
    orig_ndim = X.ndim
    if alg_axes is not None:
        X = move_alg_axes_last(X, alg_axes)
    N = X.ndim - K.n_axes
    if N < 1:
        raise ValueError("a tensor over K needs at least one mode")
    dims = X.shape[:N]
    groups = K.forward(X)
    _check_commutative(K, groups)
    # block tensors (B, n_1, ..., n_N)
    tensors = [np.moveaxis(g.data[..., 0, 0], N, 0) for g in groups]
    total = float(K.norm2(X).sum())
    return dims, groups, tensors, total, orig_ndim


def _rank_list(ranks, N):
    if ranks is None:
        return [None] * N
    if np.isscalar(ranks):
        return [int(ranks)] * N
    ranks = [int(r) for r in ranks]
    if len(ranks) != N:
        raise ValueError(f"need {N} ranks, got {len(ranks)}")
    return ranks


@dataclass
class TuckerK:
    """A Tucker decomposition over K, in block form."""

    K: Algebra
    dims: tuple
    groups: list
    cores: list                 # per group: (B, R_1, ..., R_N), zero-padded
    factors: list               # per group: list over modes of (B, n_k, R_k), zero-padded
    counts: list                # per group: (B, N) kept ranks per block and mode
    tails: list                 # per mode: weighted discarded energy
    total_energy: float
    alg_axes: object = None
    orig_ndim: int = 0
    method: str = "HOSVD"

    def reconstruct(self) -> np.ndarray:
        N = len(self.dims)
        new = []
        for g, C, Us in zip(self.groups, self.cores, self.factors):
            T = C
            for k in range(N):
                T = mode_product(T, Us[k], k)
            data = np.moveaxis(T, 0, N)[..., None, None]
            if g.division == "R":
                data = np.real(data)
            new.append(g.with_data(data))
        X = self.K.inverse(new)
        if self.alg_axes is not None:
            X = move_alg_axes_back(X, self.alg_axes, self.orig_ndim)
        return X

    def n_params(self) -> float:
        total = 0.0
        for g, c in zip(self.groups, self.counts):
            core = np.prod(c, axis=1)
            fac = c @ np.asarray(self.dims)
            per_block = np.where(core > 0, core + fac, 0)
            total += float(per_block.sum()) * g.real_params_per_entry
        return total

    @property
    def ranks_K(self) -> tuple:
        """Largest kept rank per mode over all blocks (the K-ranks for uniform truncations)."""
        return tuple(int(max(c[:, k].max() for c in self.counts)) for k in range(len(self.dims)))

    @property
    def error_bound2(self) -> float:
        return float(sum(self.tails))


@dataclass
class HOSVDK:
    """Mode-wise SVDs of a tensor over a commutative K.  Create with ``hosvd_K``."""

    K: Algebra
    dims: tuple
    groups: list
    tensors: list
    U: list          # per group: list over modes of (B, n_k, r_k)
    S: list          # per group: list over modes of (B, r_k)
    total_energy: float
    alg_axes: object = None
    orig_ndim: int = 0

    def spectra(self, k: int) -> list:
        return [S[k] for S in self.S]

    def truncate(self, ranks=None, eps=None, adaptive=False) -> TuckerK:
        N = len(self.dims)
        rl = _rank_list(ranks, N)
        delta2 = None if eps is None else float(eps) ** 2 * self.total_energy / N
        if ranks is not None and eps is not None:
            raise ValueError("give ranks or eps, not both")
        counts_by_mode, tails = [], []
        for k in range(N):
            spec = self.spectra(k)
            c = select_counts(spec, self.groups, rank=rl[k], delta2=delta2, adaptive=adaptive)
            counts_by_mode.append(c)
            tails.append(tail_energy(spec, self.groups, c))
        cores, factors, counts = [], [], []
        for gi, (g, T) in enumerate(zip(self.groups, self.tensors)):
            Us = [masked(self.U[gi][k], counts_by_mode[k][gi], g) for k in range(N)]
            C = T
            for k in range(N):
                C = mode_product(C, np.conj(np.swapaxes(Us[k], -1, -2)), k)
            cores.append(C)
            factors.append(Us)
            counts.append(np.stack([counts_by_mode[k][gi] for k in range(N)], axis=1))
        return TuckerK(self.K, self.dims, self.groups, cores, factors, counts, tails,
                       self.total_energy, self.alg_axes, self.orig_ndim, "HOSVD")


def hosvd_K(X: np.ndarray, K: Algebra, alg_axes=None, allow_noncommutative=False):
    """Mode-wise SVDs (the expensive part of the HOSVD); truncate with ``.truncate``.

    For a noncommutative K this raises NoncommutativeTuckerError unless
    allow_noncommutative=True, which returns tucker_nc.HOSVDKNC (mirror-order core, no a
    priori error bound; see tucker_nc.py)."""
    if allow_noncommutative and not K.is_commutative:
        from .tucker_nc import hosvd_nc
        return hosvd_nc(X, K, alg_axes)
    dims, groups, tensors, total, orig_ndim = _prepare(X, K, alg_axes)
    N = len(dims)
    Us, Ss = [], []
    for g, T in zip(groups, tensors):
        ug, sg = [], []
        for k in range(N):
            Mk = np.moveaxis(T, k + 1, 1).reshape(T.shape[0], dims[k], -1)
            u, s = left_singular(Mk, g)
            ug.append(u)
            sg.append(s)
        Us.append(ug)
        Ss.append(sg)
    return HOSVDK(K, dims, groups, tensors, Us, Ss, total, alg_axes, orig_ndim)


def sthosvd_K(X: np.ndarray, K: Algebra, ranks=None, eps=None, adaptive=False, order=None,
              alg_axes=None, allow_noncommutative=False, gauge=None):
    """Sequentially truncated HOSVD over a commutative K.

    ``order`` is the order in which the modes are processed (default 0, 1, ..., N-1).
    For a noncommutative K this raises NoncommutativeTuckerError unless
    allow_noncommutative=True, which runs tucker_nc.sthosvd_nc: modes processed in
    ``order`` (default N-1, ..., 0), reconstruction in the reverse order, and
    ||X - X_hat||^2 = sum_k tau_k for any K (with eps: ||X - X_hat|| <= eps ||X||).  ``gauge`` (a
    numpy Generator) multiplies every kept K-column by a random unit element of K
    (noncommutative path only; for testing the dependence on the singular vectors).
    """
    if allow_noncommutative and not K.is_commutative:
        from .tucker_nc import sthosvd_nc
        return sthosvd_nc(X, K, ranks=ranks, eps=eps, adaptive=adaptive, order=order,
                          alg_axes=alg_axes, gauge=gauge)
    dims, groups, tensors, total, orig_ndim = _prepare(X, K, alg_axes)
    N = len(dims)
    if ranks is not None and eps is not None:
        raise ValueError("give ranks or eps, not both")
    rl = _rank_list(ranks, N)
    order = list(range(N)) if order is None else [int(k) for k in order]
    if sorted(order) != list(range(N)):
        raise ValueError(f"order must be a permutation of 0..{N - 1}")
    delta2 = None if eps is None else float(eps) ** 2 * total / N
    cores = list(tensors)
    factors = [[None] * N for _ in groups]
    counts = [np.zeros((g.n_blocks, N), dtype=int) for g in groups]
    tails = [0.0] * N
    for k in order:
        Us, spec = [], []
        for g, C in zip(groups, cores):
            Mk = np.moveaxis(C, k + 1, 1).reshape(C.shape[0], C.shape[k + 1], -1)
            u, s = left_singular(Mk, g)
            Us.append(u)
            spec.append(s)
        c = select_counts(spec, groups, rank=rl[k], delta2=delta2, adaptive=adaptive)
        tails[k] = tail_energy(spec, groups, c)
        for gi, g in enumerate(groups):
            Uk = masked(Us[gi], c[gi], g)
            factors[gi][k] = Uk
            counts[gi][:, k] = c[gi]
            cores[gi] = mode_product(cores[gi], np.conj(np.swapaxes(Uk, -1, -2)), k)
    return TuckerK(K, dims, groups, cores, factors, counts, tails, total, alg_axes, orig_ndim,
                   "ST-HOSVD")
