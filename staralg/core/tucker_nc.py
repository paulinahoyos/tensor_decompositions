"""Tucker decompositions over a noncommutative K (opt-in; see hosvd_K / sthosvd_K).

Convention.  A sequence of mode products is applied from left to right, and every product
multiplies the entries of the tensor from the left:
    C x_1 U_1 x_2 ... x_d U_d := (...((C x_1 U_1) x_2 U_2) ...) x_d U_d,
    entrywise  sum_j U_d(i_d, j_d) ... U_1(i_1, j_1) C(j_1, ..., j_d).
The core is built in the MIRROR order of the reconstruction, C = A x_d U_d^H ... x_1 U_1^H
(more generally: the reconstruction applies the modes in the reverse of the order in which
the core was formed), so that every U_k meets its U_k^H:
    T = sum U_d ... U_2 (U_1 U_1^H) U_2^H ... U_d^H A.

Error identity.  Write C_{d+1} = A, C_k = C_{k+1} x_k U_k^H (in the processing order) and
tau~_k = ||C_{k+1} - C_{k+1} x_k U_k U_k^H||^2 (the energy removed at step k).  For ANY K and
any factors with orthonormal columns over K,
    ||A - T||^2 = sum_k tau~_k.
Proof: A - T telescopes into d terms (same-mode rule), D_k = (C_{k+1} - C_{k+1} x_k P_k)
x_{k+1} U_{k+1} ... x_d U_d with ||D_k||^2 = tau~_k (each x_j U_j is an isometry).  For k < j,
peel off the common isometries x_{j+1} ... x_d: the remaining mode-j unfolding of D_k is
U_j M (its last product is x_j U_j), that of D_j is (I - U_j U_j^H) C_{j+1,(j)}, and
U_j^H (I - U_j U_j^H) = 0, so the terms are orthogonal.  Only associativity, U_j^H U_j = I
and the adjoint identities are used, not commutativity.
    ST-HOSVD: U_k comes from the current core, so tau~_k is exactly the tail it truncated:
              ||A - T||^2 = sum of the truncated tails, as for real ST-HOSVD.  With a
              tolerance eps, every mode may discard eps^2 ||A||^2 / d, so ||A - T|| <= eps ||A||.
    HOSVD:    U_k comes from A_(k); for commutative K, tau~_k <= tau_k (the tail of A_(k)), which
              gives the classical bound; for noncommutative K there is no such inequality
              (left scalars from the other modes do not preserve the right column space of
              A_(k)), so there is no a priori bound, and the result depends on the choice of
              singular vectors (see ``gauge``).  The identity above still gives the error
              exactly, a posteriori.

Block form.  A block of size m (an m x m realization of M_d(D)) stores a tensor over K as
(B, n_1, ..., n_N, m, m): every entry is an m x m matrix.  The mode-k unfolding has rows
(i_k, a) and columns (other modes, c); a factor U_k is realized as (B, n_k m, r_k m), and a
mode product multiplies (i_k, a) from the left.  All modes act on the same row index a of
the entries, so factors keep whole K-columns: per-block sizes are integer K-ranks r_{k,b},
i.e. multiples of m singular values (adaptive truncation discards units of m).

Storage per block: p_rho m^2 (prod_k r_{k,b} + sum_k n_k r_{k,b}) real parameters, with p_rho
the real parameters per stored entry (R: 1, C: 2, H: 1); for m = 1 this is the commutative
count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..algebras.base import Algebra, move_alg_axes_back, move_alg_axes_last
from .blockops import left_singular, select_counts, tail_energy


# ---------------------------------------------------------------------------
# Block-tensor helpers: T has shape (B, n_1, ..., n_N, m, m)
# ---------------------------------------------------------------------------
def _perm(N, k):
    return [0, k + 1, N + 1] + [j + 1 for j in range(N) if j != k] + [N + 2]


def unfold_nc(T: np.ndarray, k: int) -> np.ndarray:
    """Realized mode-k unfolding: rows (i_k, a), columns (other modes, c)."""
    N = T.ndim - 3
    Tp = T.transpose(_perm(N, k))
    return Tp.reshape(T.shape[0], T.shape[k + 1] * T.shape[-1], -1)


def ttm_nc(T: np.ndarray, M: np.ndarray, k: int) -> np.ndarray:
    """Mode-k product with a realized matrix over K: M (B, p m, n_k m) acts from the left."""
    N = T.ndim - 3
    m = T.shape[-1]
    p = M.shape[1] // m
    Y = M @ unfold_nc(T, k)                                   # (B, p m, L)
    rest = [T.shape[j + 1] for j in range(N) if j != k]
    Y = Y.reshape((T.shape[0], p, m) + tuple(rest) + (m,))
    return Y.transpose(np.argsort(_perm(N, k)))


def _wnorm2(T: np.ndarray, g) -> float:
    """Weighted squared norm sum_b c_b ||T_b||_F^2 (the coefficient-form norm)."""
    per_block = (np.abs(T.reshape(T.shape[0], -1)) ** 2).sum(axis=1)
    return float(np.sum(g.weights * per_block))


def _random_units(rng, g, B: int, R: int) -> np.ndarray:
    """(B, R, m, m) realizations of random unit elements of the block's algebra
    (orthogonal for R, unitary for C, unitary with quaternion structure for H)."""
    m = g.m
    out = np.empty((B, R, m, m), dtype=float if g.division == "R" else complex)
    for b in range(B):
        for r in range(R):
            if g.division == "R":
                X = rng.standard_normal((m, m))
            elif g.division == "C":
                X = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
            else:
                d = m // 2
                Z = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
                W = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
                X = np.zeros((m, m), dtype=complex)
                X[0::2, 0::2], X[0::2, 1::2] = Z, W
                X[1::2, 0::2], X[1::2, 1::2] = -np.conj(W), np.conj(Z)
            u, _, vh = np.linalg.svd(X)
            out[b, r] = u @ vh                                   # polar factor
    return out


def _factor(U: np.ndarray, kranks: np.ndarray, g, gauge=None) -> np.ndarray:
    """Keep the first max(kranks) K-columns of U (B, n m, *), zero the K-columns beyond each
    block's K-rank, and optionally multiply every K-column by a random unit element."""
    m = g.m
    R = max(int(kranks.max()), 1)
    R = min(R, U.shape[-1] // m)
    F = U[..., : R * m].copy()
    if gauge is not None:
        Q = _random_units(gauge, g, F.shape[0], R)
        F4 = F.reshape(F.shape[0], F.shape[1], R, m)
        F = np.einsum("bxra,brac->bxrc", F4, Q).reshape(F.shape)
    keep = np.repeat(np.arange(R)[None, :] < kranks[:, None], m, axis=1)   # (B, R m)
    return F * keep[:, None, :]


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
    tensors = [np.moveaxis(g.data, N, 0) for g in groups]     # (B, n_1, ..., n_N, m, m)
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


def _order(order, N):
    order = list(range(N - 1, -1, -1)) if order is None else [int(k) for k in order]
    if sorted(order) != list(range(N)):
        raise ValueError(f"order must be a permutation of 0..{N - 1}")
    return order


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------
@dataclass
class TuckerKNC:
    """A Tucker decomposition over a (possibly noncommutative) K, in block form."""

    K: Algebra
    dims: tuple
    groups: list
    cores: list            # per group: (B, R_1, ..., R_N, m, m), zero-padded
    factors: list          # per group: list over modes of (B, n_k m, R_k m), zero-padded
    kranks: list           # per group: (B, N) kept K-ranks per block and mode
    tails: list            # per mode: tail of the unfolding the factor was computed from
    seq_tails: list        # per mode: energy removed when the core was formed; sum = error^2
    total_energy: float
    recon_order: list      # modes in the order the reconstruction applies them
    alg_axes: object = None
    orig_ndim: int = 0
    method: str = "ST-HOSVD"
    _shape_info: dict = field(default_factory=dict, repr=False)

    def reconstruct(self) -> np.ndarray:
        N = len(self.dims)
        new = []
        for g, C, Us in zip(self.groups, self.cores, self.factors):
            T = C
            for k in self.recon_order:
                T = ttm_nc(T, Us[k], k)
            data = np.moveaxis(T, 0, N)
            if g.division == "R":
                data = np.real(data)
            new.append(g.with_data(data))
        X = self.K.inverse(new)
        if self.alg_axes is not None:
            X = move_alg_axes_back(X, self.alg_axes, self.orig_ndim)
        return X

    def n_params(self) -> float:
        total = 0.0
        dims = np.asarray(self.dims)
        for g, r in zip(self.groups, self.kranks):
            core = np.prod(r, axis=1)
            fac = r @ dims
            per_block = np.where(core > 0, core + fac, 0)
            total += float(per_block.sum()) * g.real_params_per_entry * g.m ** 2
        return total

    @property
    def ranks_K(self) -> tuple:
        """Largest kept K-rank per mode over all blocks."""
        return tuple(int(max(r[:, k].max() for r in self.kranks)) for k in range(len(self.dims)))

    @property
    def error_apriori(self) -> float:
        """sqrt(sum_k tau_k), tau_k the tails of the unfoldings the factors were computed from.
        Equals ||A - T|| for ST-HOSVD over any K; for HOSVD it is a lower bound when K is
        commutative and neither a lower nor an upper bound in general."""
        return float(np.sqrt(max(sum(self.tails), 0.0)))

    @property
    def error_aposteriori(self) -> float:
        """sqrt(sum_k tau~_k) = ||A - T|| exactly, for HOSVD and ST-HOSVD over any K."""
        return float(np.sqrt(max(sum(self.seq_tails), 0.0)))


def _form_core(tensors, groups, factors, order):
    """Core in the given processing order, C <- C x_k U_k^H, recording the removed energy."""
    N = tensors[0].ndim - 3
    cores = list(tensors)
    seq = [0.0] * N
    for k in order:
        before = sum(_wnorm2(C, g) for C, g in zip(cores, groups))
        cores = [ttm_nc(C, np.conj(np.swapaxes(Us[k], -1, -2)), k) for C, Us, g in zip(cores, factors, groups)]
        after = sum(_wnorm2(C, g) for C, g in zip(cores, groups))
        seq[k] = max(before - after, 0.0)       # ||X||^2 - ||U^H X||^2 = ||(I - U U^H) X||^2
    return cores, seq


@dataclass
class HOSVDKNC:
    """Mode-wise SVDs of the original tensor over K.  Create with hosvd_K(..., allow_noncommutative=True)."""

    K: Algebra
    dims: tuple
    groups: list
    tensors: list
    U: list
    S: list
    total_energy: float
    alg_axes: object = None
    orig_ndim: int = 0

    def truncate(self, ranks=None, eps=None, adaptive=False, order=None, gauge=None) -> TuckerKNC:
        """ranks: uniform K-ranks; eps: every mode may discard eps^2 ||A||^2 / d of the tails of
        A (no a priori guarantee for a noncommutative K); order: the order in which the core is
        formed (default d, ..., 1); gauge: a numpy Generator to multiply every kept
        K-column by a random unit element of K."""
        N = len(self.dims)
        if ranks is not None and eps is not None:
            raise ValueError("give ranks or eps, not both")
        rl = _rank_list(ranks, N)
        delta2 = None if eps is None else float(eps) ** 2 * self.total_energy / N
        units = [g.m for g in self.groups]
        tails, kr_by_mode = [], []
        for k in range(N):
            spec = [S[k] for S in self.S]
            c = select_counts(spec, self.groups, rank=rl[k], delta2=delta2, adaptive=adaptive, units=units)
            tails.append(tail_energy(spec, self.groups, c))
            kr_by_mode.append([ci // g.m for ci, g in zip(c, self.groups)])
        factors = [[_factor(self.U[gi][k], kr_by_mode[k][gi], g, gauge) for k in range(N)]
                   for gi, g in enumerate(self.groups)]
        order = _order(order, N)
        cores, seq = _form_core(self.tensors, self.groups, factors, order)
        kranks = [np.stack([kr_by_mode[k][gi] for k in range(N)], axis=1) for gi in range(len(self.groups))]
        return TuckerKNC(self.K, self.dims, self.groups, cores, factors, kranks, tails, seq,
                         self.total_energy, order[::-1], self.alg_axes, self.orig_ndim, "HOSVD")


def hosvd_nc(X, K, alg_axes=None) -> HOSVDKNC:
    dims, groups, tensors, total, orig_ndim = _prepare(X, K, alg_axes)
    N = len(dims)
    Us, Ss = [], []
    for g, T in zip(groups, tensors):
        ug, sg = [], []
        for k in range(N):
            u, s = left_singular(unfold_nc(T, k), g)
            ug.append(u)
            sg.append(s)
        Us.append(ug)
        Ss.append(sg)
    return HOSVDKNC(K, dims, groups, tensors, Us, Ss, total, alg_axes, orig_ndim)


def sthosvd_nc(X, K, ranks=None, eps=None, adaptive=False, order=None, alg_axes=None,
               gauge=None) -> TuckerKNC:
    """ST-HOSVD over any K, processing the modes in ``order`` (default d, ..., 1) and
    reconstructing in the reverse order.  ||A - T||^2 = sum of the truncated tails for any K;
    with eps, every mode may discard eps^2 ||A||^2 / d, so ||A - T|| <= eps ||A||."""
    dims, groups, tensors, total, orig_ndim = _prepare(X, K, alg_axes)
    N = len(dims)
    if ranks is not None and eps is not None:
        raise ValueError("give ranks or eps, not both")
    rl = _rank_list(ranks, N)
    order = _order(order, N)
    delta2 = None if eps is None else float(eps) ** 2 * total / N
    units = [g.m for g in groups]
    cores = list(tensors)
    factors = [[None] * N for _ in groups]
    kranks = [np.zeros((g.n_blocks, N), dtype=int) for g in groups]
    tails = [0.0] * N
    seq = [0.0] * N
    for k in order:
        Us, spec = [], []
        for g, C in zip(groups, cores):
            u, s = left_singular(unfold_nc(C, k), g)
            Us.append(u)
            spec.append(s)
        c = select_counts(spec, groups, rank=rl[k], delta2=delta2, adaptive=adaptive, units=units)
        tails[k] = tail_energy(spec, groups, c)
        before = sum(_wnorm2(C, g) for C, g in zip(cores, groups))
        for gi, g in enumerate(groups):
            kr = c[gi] // g.m
            F = _factor(Us[gi], kr, g, gauge)
            factors[gi][k] = F
            kranks[gi][:, k] = kr
            cores[gi] = ttm_nc(cores[gi], np.conj(np.swapaxes(F, -1, -2)), k)
        after = sum(_wnorm2(C, g) for C, g in zip(cores, groups))
        seq[k] = max(before - after, 0.0)
    return TuckerKNC(K, dims, groups, cores, factors, kranks, tails, seq, total, order[::-1],
                     alg_axes, orig_ndim, "ST-HOSVD")
