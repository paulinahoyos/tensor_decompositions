"""SVD of matrices over a *-algebra K (Algorithm 1) and its truncations.

    svd = svd_K(A, K, alg_axes=...)       # full (economy) SVD, computed blockwise
    Ak  = svd.reconstruct(rank=k)          # uniform truncation: K-rank k (Definition 3.14)
    Ak  = svd.reconstruct(energy=0.99)     # adaptive: per-block ranks keeping 99% energy
    Ak  = svd.reconstruct(budget=5000)     # adaptive: best energy within a parameter budget
    U, S, V = svd.factors(k)               # U, Sigma, V as matrices over K (coefficient form)

Uniform truncation keeps k d_rho singular values over D_rho in block rho (k s
stored singular values, s the stored block size), which is the rank-k truncation
of the ordered SVD returned by Algorithm 1; by Theorem 4.2 it is optimal among
matrices of K-rank <= k.  Adaptive truncation chooses a separate rank for every
block (as in the t-SVDII of Kilmer, Horesh, Avron and Newman); the error formula
||A - A_hat||^2 = sum_rho c_rho sum_{discarded} sigma^2 holds for both.

Storage of a truncation is counted as the real parameters of U and of S V^H:
(m s + n s) real-parameters-per-entry for every stored singular triplet.  For a
uniform K-rank k this equals k (m + n) dim_R K.

Quaternion blocks (any size d).  A block in M_d(H) is stored as a 2d x 2d complex
matrix, realized entry by entry with chi, so the rows and columns of every block
matrix come in consecutive pairs (index (i, a), a in {0, 1}) and
conj(X) = J X J^{-1} with J = I (x) [[0, 1], [-1, 0]].  numpy's complex SVD returns
singular values in equal pairs but an arbitrary basis of each 2-dimensional singular
space; ``_quaternion_fix`` rebuilds the bases as pairs (w, J^{-1} conj(w)), so that U
and V map back to quaternion matrices.  Only the pairing of consecutive rows is used,
so the same code serves every d.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..algebras.base import Algebra, Group, move_alg_axes_back, move_alg_axes_last
from .kmatrix import blocks_to_groups, from_blocks, to_blocks


# ---------------------------------------------------------------------------
# Quaternion structure
# ---------------------------------------------------------------------------
def _jconj(x: np.ndarray) -> np.ndarray:
    """J^{-1} conj(x) on vectors whose rows are ordered (i, a), a in {0, 1}.

    For a block matrix X of 2x2 quaternion realizations, conj(X) = J X J^{-1} with
    J = I (x) [[0, 1], [-1, 0]], so (u, v) -> (jconj(u), jconj(v)) maps singular
    pairs to singular pairs.
    """
    y = np.empty_like(x)
    y[0::2] = -np.conj(x[1::2])
    y[1::2] = np.conj(x[0::2])
    return y


def _gs_pairs_into(Q, cnt, candidates, need):
    """Append ``need`` columns to Q[:, :cnt] as pairs (w, jconj(w)), orthonormal to all
    previous columns, taken from ``candidates`` (an iterable of vectors).  Returns the new
    column count.  Q is preallocated, so no basis is restacked."""
    target = cnt + need
    for c in candidates:
        if cnt >= target:
            break
        w = np.array(c, dtype=complex)
        for _ in range(2):
            if cnt:
                P = Q[:, :cnt]
                w = w - P @ (P.conj().T @ w)
        nw = np.linalg.norm(w)
        if nw < 1e-6:
            continue
        w = w / nw
        Q[:, cnt] = w
        Q[:, cnt + 1] = _jconj(w)
        cnt += 2
    if cnt < target:
        raise RuntimeError("could not complete a quaternionic orthonormal basis")
    return cnt


def _candidates(rest_cols, n):
    """Remaining columns of an SVD factor, then the standard basis (generated lazily)."""
    yield from rest_cols.T
    for i in range(n):
        e = np.zeros(n, dtype=complex)
        e[i] = 1.0
        yield e


def _structure_pairs(u, s, rtol=1e-10, cluster_rtol=1e-7):
    """Rebuild the left singular bases of quaternion blocks as pairs (w, J^{-1} conj(w)).

    u: (B, 2m, r), s: (B, r) from a complex SVD (or eigendecomposition) of quaternion
    block matrices.  Each nonzero singular value of a quaternion matrix appears as an
    equal pair whose singular space is J-invariant.  For an isolated pair with first
    vector w, J^{-1} conj(w) lies in the same space and is orthogonal to w, so the
    structured basis is (w, J^{-1} conj(w)) with no orthogonalization.  Clusters of
    more than two (near-)equal singular values and the null space are completed by
    quaternionic Gram-Schmidt.  Singular values of a cluster are averaged.
    Returns (U, s, nz) with nz the number of nonzero singular values per block.
    """
    u = u.astype(complex, copy=True)
    s = s.astype(np.float64, copy=True)
    B, M2, r = u.shape
    nzs = np.zeros(B, dtype=int)
    for b in range(B):
        sb = s[b]
        smax = sb[0] if r else 0.0
        nz = int(np.sum(sb > rtol * max(smax, np.finfo(float).tiny)))
        nz -= nz % 2
        Q = np.empty((M2, r), dtype=complex)
        cnt = 0
        i = 0
        while i < nz:
            j = i + 1
            while j < nz and abs(sb[j] - sb[i]) <= cluster_rtol * max(smax, 1.0):
                j += 1
            if (j - i) % 2:
                j = min(j + 1, nz)
            if j - i == 2:                       # isolated pair: no orthogonalization needed
                w = u[b][:, i] / np.linalg.norm(u[b][:, i])
                Q[:, cnt] = w
                Q[:, cnt + 1] = _jconj(w)
                cnt += 2
            else:
                cnt = _gs_pairs_into(Q, cnt, u[b][:, i:j].T, j - i)
            sb[i:j] = sb[i:j].mean()
            i = j
        if cnt < r:
            cnt = _gs_pairs_into(Q, cnt, _candidates(u[b][:, cnt:], M2), r - cnt)
        u[b] = Q
        nzs[b] = nz
    return u, s, nzs


def _quaternion_fix(X, u, s, v, rtol=1e-10, cluster_rtol=1e-7):
    """Make the SVD of quaternion blocks structured.  X: (B, 2m, 2n); u, s, v from numpy."""
    U, s, nzs = _structure_pairs(u, s, rtol, cluster_rtol)
    B, M2, r = U.shape
    N2 = v.shape[1]
    V = np.zeros((B, N2, r), dtype=complex)
    for b in range(B):
        nz = nzs[b]
        if nz:
            V[b][:, :nz] = X[b].conj().T @ U[b][:, :nz] / s[b, :nz]
        if r - nz:
            _gs_pairs_into(V[b], nz, _candidates(v[b][:, nz:], N2), r - nz)
    return U, s, V


def _block_svd(M: np.ndarray, g: Group):
    """Economy SVD of every block in a group.  Returns U (B, ms, r), S (B, r), V (B, ns, r)."""
    u, s, vh = np.linalg.svd(M, full_matrices=False)
    v = np.conj(np.swapaxes(vh, -1, -2))
    if g.division == "H":
        u, s, v = _quaternion_fix(M, u, s, v)
    return u, s, v


# ---------------------------------------------------------------------------
# The SVD object
# ---------------------------------------------------------------------------
@dataclass
class SVDK:
    """Blockwise SVD of a matrix over K.  Create it with ``svd_K``."""

    K: Algebra
    m: int
    n: int
    templates: list
    U: list
    S: list
    V: list
    alg_axes: object = None
    orig_ndim: int = 0
    _total: float = field(default=0.0, repr=False)

    # ---- per-block bookkeeping ------------------------------------------------
    def _units(self, g: Group) -> int:
        """Singular values kept together: pairs for quaternion blocks."""
        return 2 if g.division == "H" else 1

    def _cost_per_triplet(self, g: Group) -> float:
        s = g.m
        return (self.m * s + self.n * s) * g.real_params_per_entry

    @property
    def total_energy(self) -> float:
        return self._total

    # ---- truncation masks -------------------------------------------------------
    def masks(self, rank=None, energy=None, budget=None) -> list[np.ndarray]:
        """Boolean masks (B, r) of kept singular values for each group."""
        given = sum(x is not None for x in (rank, energy, budget))
        if given > 1:
            raise ValueError("give at most one of rank, energy, budget")
        if given == 0:
            return [np.ones(S.shape, dtype=bool) for S in self.S]
        if rank is not None:
            out = []
            for g, S in zip(self.templates, self.S):
                keep = min(int(rank) * g.m, S.shape[-1])
                mk = np.zeros(S.shape, dtype=bool)
                mk[:, :keep] = True
                out.append(mk)
            return out
        # adaptive: rank units of all blocks by energy (or energy per parameter)
        units = []  # (score, energy, cost, group, block, unit_index)
        for gi, (g, S) in enumerate(zip(self.templates, self.S)):
            u = self._units(g)
            cost = u * self._cost_per_triplet(g)
            B, r = S.shape
            for b in range(B):
                for t in range(r // u):
                    e = g.weights[b] * float(np.sum(S[b, t * u:(t + 1) * u] ** 2))
                    score = e if energy is not None else e / cost
                    units.append((score, e, cost, gi, b, t))
        units.sort(key=lambda z: -z[0])
        counts = [np.zeros(S.shape[0], dtype=int) for S in self.S]
        if energy is not None:
            target = float(energy) * self._total
            acc = 0.0
            for _, e, _, gi, b, t in units:
                if acc >= target * (1 - 1e-12):
                    break
                counts[gi][b] += 1
                acc += e
        else:
            spent = 0.0
            for _, e, cost, gi, b, t in units:
                if spent + cost <= budget:
                    counts[gi][b] += 1
                    spent += cost
        out = []
        for g, S, c in zip(self.templates, self.S, counts):
            mk = np.zeros(S.shape, dtype=bool)
            u = self._units(g)
            for b, cb in enumerate(c):
                mk[b, :cb * u] = True
            out.append(mk)
        return out

    # ---- results ------------------------------------------------------------------
    def reconstruct(self, rank=None, energy=None, budget=None) -> np.ndarray:
        """Truncated approximation in coefficient form (same layout as the input)."""
        mats = []
        for U, S, V, mk in zip(self.U, self.S, self.V, self.masks(rank, energy, budget)):
            mats.append((U * (S * mk)[:, None, :]) @ np.conj(np.swapaxes(V, -1, -2)))
        A = from_blocks(mats, self.templates, self.K, self.m, self.n)
        if self.alg_axes is not None:
            A = move_alg_axes_back(A, self.alg_axes, self.orig_ndim)
        return A

    def n_params(self, rank=None, energy=None, budget=None) -> float:
        """Real parameters stored by the truncation (U and S V^H)."""
        total = 0.0
        for g, mk in zip(self.templates, self.masks(rank, energy, budget)):
            total += mk.sum() * self._cost_per_triplet(g)
        return total

    def tail_energy(self, rank=None, energy=None, budget=None) -> float:
        """sum_rho c_rho sum_{discarded} sigma^2  (= ||A - A_hat||^2)."""
        tail = 0.0
        for g, S, mk in zip(self.templates, self.S, self.masks(rank, energy, budget)):
            tail += float(np.sum(g.weights[:, None] * (S ** 2) * (~mk)))
        return tail

    def factors(self, k: int | None = None):
        """U (m x k), Sigma (k x k), V (n x k) over K for the uniform truncation of rank k.

        Returned in coefficient form with the algebra axes last.  A = U Sigma V^H when
        k = min(m, n).
        """
        kmax = min(self.m, self.n)
        k = kmax if k is None else int(k)
        if not 1 <= k <= kmax:
            raise ValueError(f"k must be between 1 and {kmax}")
        Um, Sm, Vm = [], [], []
        for g, U, S, V in zip(self.templates, self.U, self.S, self.V):
            q = k * g.m
            Um.append(U[:, :, :q])
            Vm.append(V[:, :, :q])
            D = np.zeros(S.shape[:1] + (q, q), dtype=U.dtype)
            idx = np.arange(q)
            D[:, idx, idx] = S[:, :q]
            Sm.append(D)
        Ugroups = blocks_to_groups(Um, self.templates, self.m, k)
        Sgroups = blocks_to_groups(Sm, self.templates, k, k)
        Vgroups = blocks_to_groups(Vm, self.templates, self.n, k)
        return self.K.inverse(Ugroups), self.K.inverse(Sgroups), self.K.inverse(Vgroups)

    def singular_values(self) -> list[np.ndarray]:
        """Stored singular values of every block, per group: arrays (B, r)."""
        return [S.copy() for S in self.S]


def svd_K(A: np.ndarray, K: Algebra, alg_axes=None) -> SVDK:
    """SVD of a matrix over K (Algorithm 1).

    Parameters
    ----------
    A : array
        The matrix in coefficient form.  Without ``alg_axes``, its shape is
        (m, n, *K.shape).  With ``alg_axes``, those axes of A hold the algebra
        coordinates (in the order of K's factors) and the two remaining axes are the
        rows and columns.
    K : Algebra
    alg_axes : int or sequence of int, optional
    """
    A = np.asarray(A)
    orig_ndim = A.ndim
    if alg_axes is not None:
        A = move_alg_axes_last(A, alg_axes)
    m, n = A.shape[:2]
    mats, groups = to_blocks(A, K)
    Us, Ss, Vs = [], [], []
    for M, g in zip(mats, groups):
        u, s, v = _block_svd(M, g)
        Us.append(u)
        Ss.append(s)
        Vs.append(v)
    total = float(K.norm2(A).sum())
    return SVDK(K, m, n, groups, Us, Ss, Vs, alg_axes, orig_ndim, total)


def rank_K(A: np.ndarray, K: Algebra, alg_axes=None, rtol: float = 1e-10) -> int:
    """Numerical K-rank: max over blocks of ceil(rank(block) / s) (Proposition 3.13)."""
    svd = svd_K(A, K, alg_axes)
    smax = max((float(S.max()) for S in svd.S if S.size), default=0.0)
    if smax == 0:
        return 0
    rk = 0
    for g, S in zip(svd.templates, svd.S):
        r_stored = (S > rtol * smax).sum(axis=-1)
        rk = max(rk, int(np.ceil(r_stored.max() / g.m)))
    return rk


# ---------------------------------------------------------------------------
# Leading left singular vectors (Algorithm 2)
# ---------------------------------------------------------------------------
def llsv_blocks(mats: list[np.ndarray], templates: list[Group], k: int) -> list[np.ndarray]:
    """Leading k s left singular vectors of every block: list of (B, m s, k s) arrays.

    Wide blocks (many more columns than rows) use the eigenvectors of M M^H, which is
    much cheaper in memory; quaternion blocks always use the structured SVD.
    """
    out = []
    for M, g in zip(mats, templates):
        q = min(k * g.m, M.shape[-2])
        if g.division == "H":
            u, _, _ = _block_svd(M, g)
            out.append(u[:, :, :q])
            continue
        rows, cols = M.shape[-2:]
        if cols > 2 * rows:
            G = M @ np.conj(np.swapaxes(M, -1, -2))
            w, V = np.linalg.eigh(G)
            out.append(V[:, :, ::-1][:, :, :q])
        else:
            u, _, _ = np.linalg.svd(M, full_matrices=False)
            out.append(u[:, :, :q])
    return out


def llsv_K(A: np.ndarray, K: Algebra, k: int, alg_axes=None) -> np.ndarray:
    """Leading k left singular vectors of A over K, as an m x k matrix over K.

    U U^H A is the rank-k truncation of an ordered SVD of A (Eq. 24).
    The result has the algebra axes last.
    """
    A = np.asarray(A)
    if alg_axes is not None:
        A = move_alg_axes_last(A, alg_axes)
    m = A.shape[0]
    mats, groups = to_blocks(A, K)
    U = llsv_blocks(mats, groups, k)
    k_eff = min(k, m)
    return K.inverse(blocks_to_groups(U, groups, m, k_eff))
