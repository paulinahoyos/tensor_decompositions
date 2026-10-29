"""Tensor-train decomposition over K (TT-SVD), valid for every K, commutative or not.

    tt = tt_svd_K(X, K, ranks=(4, 6, 4))            # uniform K-ranks r_1, ..., r_{N-1}
    tt = tt_svd_K(X, K, eps=0.1)                     # smallest uniform ranks, error <= eps ||X||
    tt = tt_svd_K(X, K, eps=0.1, adaptive=True)      # per-block ranks, error <= eps ||X||
    tt = tt_svd_K(X, K, ranks=..., order=(2, 0, 1))  # modes taken in this order
    X_hat = tt.reconstruct();  tt.n_params();  tt.tails

Algorithm.  Step k unfolds the remainder as a matrix over K with rows
(r_{k-1}, i_k) and columns (i_{k+1}, ..., i_N), takes its leading left singular
vectors U_k over K (the k-th core), and continues with U_k^H times the remainder.  The
projections are nested and every core is an isometry, so

    ||X - X_hat||^2 = sum_k tau_k,     tau_k <= best rank-r_k error^2 of the k-th unfolding,

hence ||X - X_hat|| <= sqrt(N - 1) * (best TT error).  Neither fact uses
commutativity: only products U^H A and U (U^H A) of matrices over K appear.

Block form.  In block rho (stored size s), a tensor over K with s x s entries is the
array (i_1, a, i_2, ..., i_N, c); step k works on the matrix with rows
((R_{k-1}/u, i_k), u) and columns (i_{k+1}, ..., i_N, c), where u is 2 for quaternion
blocks (keeping the pair index fastest preserves the quaternion structure) and 1
otherwise.  Row orders that differ by a permutation give the same truncation.

Storage (real parameters): for each block, params_per_entry * sum_k R_{k-1} n_k R_k with
R_0 = R_N = s; for uniform K-ranks r this is (sum_k r_{k-1} n_k r_k) dim_R K, r_0 = r_N = 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..algebras.base import Algebra, move_alg_axes_back, move_alg_axes_last
from .blockops import left_singular, masked, select_counts, tail_energy, unit


@dataclass
class TTK:
    """A tensor train over K, in block form.  Create with ``tt_svd_K``."""

    K: Algebra
    dims: tuple                 # mode sizes in the processing order
    order: tuple                # processing order (a permutation of the original modes)
    groups: list
    cores: list                 # per group: list of N-1 matrices U_k (B, rows_k, R_k), zero-padded
    last: list                  # per group: (B, rows_N, s)
    counts: list                # per group: (B, N-1) kept ranks R_k per block
    tails: list                 # per step: weighted discarded energy
    total_energy: float
    alg_axes: object = None
    orig_ndim: int = 0

    def reconstruct(self) -> np.ndarray:
        N = len(self.dims)
        new = []
        for g, Us, W in zip(self.groups, self.cores, self.last):
            u, s, B = unit(g), g.m, g.n_blocks
            for k in range(N - 2, -1, -1):
                R = Us[k].shape[-1]
                nxt = self.dims[k + 1]
                rest = W.shape[-1]
                W = W.reshape(B, R // u, nxt, u, rest).transpose(0, 1, 3, 2, 4).reshape(B, R, nxt * rest)
                W = Us[k] @ W
            # W: (B, n_1 s, n_2 ... n_N s) -> (n_1, ..., n_N, B, s, s)
            T = W.reshape((B, self.dims[0], s) + tuple(self.dims[1:]) + (s,))
            perm = [0, 1] + list(range(3, N + 2)) + [2, N + 2]   # (B, n_1, ..., n_N, a, c)
            T = np.moveaxis(T.transpose(perm), 0, N)
            if g.division == "R":
                T = np.real(T)
            new.append(g.with_data(T))
        X = self.K.inverse(new)
        # undo the mode order
        N_alg = self.K.n_axes
        inv = list(np.argsort(self.order)) + list(range(N, N + N_alg))
        X = np.transpose(X, inv)
        if self.alg_axes is not None:
            X = move_alg_axes_back(X, self.alg_axes, self.orig_ndim)
        return X

    def n_params(self) -> float:
        total = 0.0
        dims = np.asarray(self.dims)
        for g, c in zip(self.groups, self.counts):
            B = c.shape[0]
            R = np.concatenate([np.full((B, 1), g.m), c, np.full((B, 1), g.m)], axis=1)
            per_block = (R[:, :-1] * dims[None, :] * R[:, 1:]).sum(axis=1)
            per_block = np.where(c.min(axis=1) > 0, per_block, 0)
            total += float(per_block.sum()) * g.real_params_per_entry
        return total

    @property
    def ranks_K(self) -> tuple:
        """Per step, max over blocks of ceil(R_k / s): the K-ranks of a uniform truncation."""
        return tuple(int(max(np.ceil(c[:, k] / g.m).max() for g, c in zip(self.groups, self.counts)))
                     for k in range(len(self.dims) - 1))

    @property
    def error_bound2(self) -> float:
        return float(sum(self.tails))


def _rank_list(ranks, N):
    if ranks is None:
        return [None] * (N - 1)
    if np.isscalar(ranks):
        return [int(ranks)] * (N - 1)
    ranks = [int(r) for r in ranks]
    if len(ranks) != N - 1:
        raise ValueError(f"a tensor train of order {N} needs {N - 1} ranks, got {len(ranks)}")
    return ranks


def tt_svd_K(X: np.ndarray, K: Algebra, ranks=None, eps=None, adaptive=False, order=None,
             alg_axes=None) -> TTK:
    """TT-SVD of a tensor over K (any K).

    Parameters
    ----------
    X : array
        The tensor in coefficient form: (n_1, ..., n_N, *K.shape), or any layout with
        ``alg_axes`` naming the algebra axes.
    ranks : int or sequence of N-1 ints, optional
        Uniform K-ranks.
    eps : float, optional
        Relative tolerance: every step may discard eps^2 ||X||^2 / (N - 1).
    adaptive : bool
        With eps, choose the ranks separately in every block.
    order : sequence, optional
        Order in which the tensor modes are taken (a permutation of 0..N-1).
    """
    X = np.asarray(X)
    orig_ndim = X.ndim
    if alg_axes is not None:
        X = move_alg_axes_last(X, alg_axes)
    N = X.ndim - K.n_axes
    if N < 2:
        raise ValueError("a tensor train needs at least two modes")
    if ranks is not None and eps is not None:
        raise ValueError("give ranks or eps, not both")
    order = tuple(range(N)) if order is None else tuple(int(k) for k in order)
    if sorted(order) != list(range(N)):
        raise ValueError(f"order must be a permutation of 0..{N - 1}")
    X = np.transpose(X, list(order) + list(range(N, X.ndim)))
    dims = X.shape[:N]
    rl = _rank_list(ranks, N)
    total = float(K.norm2(X).sum())
    delta2 = None if eps is None else float(eps) ** 2 * total / (N - 1)

    groups = K.forward(X)
    Ws = []
    for g in groups:
        B, s = g.n_blocks, g.m
        T = np.moveaxis(g.data, N, 0)                                   # (B, n_1..n_N, a, c)
        perm = [0, 1, N + 1] + list(range(2, N + 1)) + [N + 2]          # (B, n_1, a, n_2..n_N, c)
        Ws.append(T.transpose(perm).reshape(B, dims[0] * s, -1))

    cores = [[] for _ in groups]
    counts = [[] for _ in groups]
    tails = []
    for k in range(N - 1):
        Us, spec = [], []
        for g, W in zip(groups, Ws):
            U, S = left_singular(W, g)
            Us.append(U)
            spec.append(S)
        c = select_counts(spec, groups, rank=rl[k], delta2=delta2, adaptive=adaptive)
        tails.append(tail_energy(spec, groups, c))
        for gi, g in enumerate(groups):
            u, B = unit(g), g.n_blocks
            Uk = masked(Us[gi], c[gi], g)
            W = np.conj(np.swapaxes(Uk, -1, -2)) @ Ws[gi]                # (B, R, cols)
            R = Uk.shape[-1]
            nxt = dims[k + 1]
            rest = W.shape[-1] // nxt
            Ws[gi] = W.reshape(B, R // u, u, nxt, rest).transpose(0, 1, 3, 2, 4).reshape(B, (R // u) * nxt * u, rest)
            cores[gi].append(Uk)
            counts[gi].append(c[gi])
    counts = [np.stack(c, axis=1) for c in counts]
    return TTK(K, dims, order, groups, cores, Ws, counts, tails, total, alg_axes, orig_ndim)
