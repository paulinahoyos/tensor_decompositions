"""Tree tensor network decomposition over a commutative K (TTN-SVD, Algorithm 6).

    tree = (((0,), (1, 2)), (3,), (4,))                  # Figure 3 of the paper (modes from 0)
    ttn = ttn_svd_K(X, K, tree, ranks=(2, 3, 4, 2, 2))   # K-ranks r_v, vertices in the order of T
    ttn = ttn_svd_K(X, K, tree, eps=0.1)                 # smallest uniform ranks, error <= eps ||X||
    ttn = ttn_svd_K(X, K, tree, eps=0.1, adaptive=True)  # per-block ranks, error <= eps ||X||
    X_hat = ttn.reconstruct();  ttn.n_params();  ttn.tails;  ttn.cores_K()

Trees.  A vertex is a nonempty tuple of the edges below it, from left to right: an int is
the external edge of that tensor mode, a tuple is a child vertex.  The outermost tuple is
the root v_0.  The external edges read from left to right give the order in which the
modes are drawn (any permutation of 0..N-1; the tensor is transposed accordingly).  The
order of T lists the vertices from left to right, each after its children (post-order).
``tt_tree``, ``tucker_tree`` and ``ht_tree`` build the TT path, the Tucker star and a
balanced hierarchical Tucker tree.

Algorithm.  For each vertex v != v_0 in the order of T, Y is unfolded along the internal
edge from v to its parent (rows: the edges below v, columns: all other open edges), C_v is
the leading left singular vectors over K of that unfolding, and Y <- C_v^H Y.  The root
core is what remains.  Every step is an orthogonal projection, so with tau_v the tail
discarded at vertex v and b = #V(T) - 1,

    ||X - X_hat||^2 = sum_v tau_v,     tau_v <= best rank-r_v error^2 of the unfolding of X
                                                along the edge of v,

hence ||X - X_hat|| <= sqrt(b) ||X - T*|| (Theorem 5.12).  With a tolerance eps, every step
may discard eps^2 ||X||^2 / b, so ||X - X_hat|| <= eps ||X||.

Why only commutative K.  The bound on tau_v reorders products in distinct modes, which
needs a commutative K (Appendix C).  The noncommutative case is not implemented yet:
ttn_svd_K raises NoncommutativeTTNError; tt_svd_K covers the TT path for every K.  A very
proper commutative K is isomorphic to R^a (+) C^b, so all blocks are 1 x 1 and a tensor
over K is a stack of real and complex tensors, one per block; the TTN over K is a TTN of
every block with shared (uniform) or separate (adaptive) ranks.

Storage (real parameters): for each block,
  params_per_entry * (sum_{v != v_0} R_v prod_{e below v} s_e + prod_{e below v_0} s_e);
for uniform K-ranks this is the same sum times dim_R K.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np

from ..algebras.base import Algebra, move_alg_axes_back, move_alg_axes_last
from .blockops import left_singular, masked, select_counts, tail_energy
from .tucker import NoncommutativeTuckerError


class NoncommutativeTTNError(NoncommutativeTuckerError):
    """Raised when a tree tensor network is requested over a noncommutative K."""


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------
def tt_tree(N: int) -> tuple:
    """The TT path C_1 - C_2 - ... - C_N, rooted at C_N: ((((0,), 1), 2), ..., N - 1)."""
    tree = (0,)
    for k in range(1, N):
        tree = (tree, k)
    return tree


def tucker_tree(N: int) -> tuple:
    """The Tucker star: factors U_1, ..., U_N below the core, ((0,), (1,), ..., (N - 1,))."""
    return tuple((k,) for k in range(N))


def ht_tree(N: int) -> tuple:
    """A balanced binary hierarchical Tucker tree with 2N - 1 vertices."""
    def split(lo, hi):
        if hi - lo == 1:
            return (lo,)
        mid = (lo + hi + 1) // 2
        return (split(lo, mid), split(mid, hi))
    return split(0, N)


def _parse_tree(tree):
    """Vertices in the order of T, each as its list of edge labels below it, ('x', mu) for
    the external edge of mode mu and ('v', t) for the edge to the t-th vertex; and the modes
    of the external edges from left to right."""
    vertices, leaves = [], []

    def walk(node):
        if not isinstance(node, (tuple, list)) or len(node) == 0:
            raise ValueError(f"a vertex must be a nonempty tuple of edges, got {node!r}")
        below = []
        for item in node:
            if isinstance(item, (tuple, list)):
                below.append(("v", walk(item)))
            else:
                mu = operator.index(item)
                leaves.append(mu)
                below.append(("x", mu))
        vertices.append(below)
        return len(vertices) - 1

    walk(tree)
    return vertices, leaves


# ---------------------------------------------------------------------------
# Block-tensor helpers: Y has shape (B, e_1, ..., e_p)
# ---------------------------------------------------------------------------
def _unfold(Y: np.ndarray, p0: int, k: int) -> np.ndarray:
    """Unfolding with rows (e_{p0}, ..., e_{p0+k-1}) and columns (the other edges)."""
    B = Y.shape[0]
    rows = Y.shape[1 + p0: 1 + p0 + k]
    Yp = np.moveaxis(Y, list(range(1 + p0, 1 + p0 + k)), list(range(1, 1 + k)))
    return Yp.reshape(B, int(np.prod(rows)), -1)


def _fold(Z: np.ndarray, row_shape: tuple, other_shape: tuple, p0: int) -> np.ndarray:
    """Inverse of _unfold: Z (B, prod(row_shape), prod(other_shape)) -> block tensor with the
    row edges placed at positions p0, p0 + 1, ... among the other edges."""
    k = len(row_shape)
    Y = Z.reshape((Z.shape[0],) + tuple(row_shape) + tuple(other_shape))
    return np.moveaxis(Y, list(range(1, 1 + k)), list(range(1 + p0, 1 + p0 + k)))


def _others(Y: np.ndarray, p0: int, k: int) -> tuple:
    edges = Y.shape[1:]
    return tuple(edges[:p0]) + tuple(edges[p0 + k:])


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------
@dataclass
class TTNK:
    """A tree tensor network over a commutative K, in block form.  Create with ``ttn_svd_K``."""

    K: Algebra
    tree: tuple
    dims: tuple                 # mode sizes in the drawing order (external edges left to right)
    order: tuple                # the modes of the external edges, left to right
    vertices: list              # per vertex in the order of T: edge labels below it
    groups: list
    cores: list                 # per group: list of b cores (B, M_v, R_v), zero-padded
    root: list                  # per group: (B, s_1, ..., s_k)
    counts: list                # per group: (B, b) kept ranks per block and vertex v != v_0
    tails: list                 # per vertex v != v_0: weighted discarded energy
    total_energy: float
    alg_axes: object = None
    orig_ndim: int = 0

    def _sizes(self, gi: int, below, ranks=None) -> tuple:
        """Sizes of the edges below a vertex: the stored ones of group gi, or ``ranks`` for
        the internal edges when given."""
        return tuple(self.dims[mu] if kind == "x" else
                     (ranks[mu] if ranks is not None else self.cores[gi][mu].shape[-1])
                     for kind, mu in below)

    def reconstruct(self) -> np.ndarray:
        N, b = len(self.dims), len(self.vertices) - 1
        new = []
        for gi, g in enumerate(self.groups):
            Y = self.root[gi]
            edges = list(self.vertices[-1])
            for t in range(b - 1, -1, -1):
                p0 = edges.index(("v", t))
                other = _others(Y, p0, 1)
                Z = self.cores[gi][t] @ _unfold(Y, p0, 1)
                Y = _fold(Z, self._sizes(gi, self.vertices[t]), other, p0)
                edges[p0:p0 + 1] = self.vertices[t]
            data = np.moveaxis(Y, 0, N)[..., None, None]
            if g.division == "R":
                data = np.real(data)
            new.append(g.with_data(data))
        X = self.K.inverse(new)
        inv = list(np.argsort(self.order)) + list(range(N, N + self.K.n_axes))
        X = np.transpose(X, inv)
        if self.alg_axes is not None:
            X = move_alg_axes_back(X, self.alg_axes, self.orig_ndim)
        return X

    def cores_K(self) -> list:
        """The cores in coefficient form, in the order of T: C_v of shape
        (s_1, ..., s_k, r_v, *K.shape) for v != v_0, and (s_1, ..., s_k, *K.shape) for the
        root, where s_1, ..., s_k are the edges below v (external ones in the drawing order,
        internal ones of size ranks_K).  Blocks with a smaller rank are zero-padded."""
        rk = self.ranks_K
        b = len(rk)
        out = []
        for t, below in enumerate(self.vertices):
            new = []
            for gi, g in enumerate(self.groups):
                rows = self._sizes(gi, below)
                if t < b:
                    C = self.cores[gi][t]
                    C = C.reshape((C.shape[0],) + rows + (C.shape[-1],))
                    shape = self._sizes(gi, below, rk) + (rk[t],)
                else:
                    C = self.root[gi]
                    shape = self._sizes(gi, below, rk)
                C = np.moveaxis(C, 0, -1)                                # (rows, [R], B)
                C = np.pad(C, [(0, s - n) for s, n in zip(shape, C.shape)] + [(0, 0)])
                if g.division == "R":
                    C = np.real(C)
                new.append(g.with_data(C[..., None, None]))
            out.append(self.K.inverse(new))
        return out

    def n_params(self) -> float:
        total = 0.0
        b = len(self.vertices) - 1
        for g, c in zip(self.groups, self.counts):
            def size(e):
                kind, mu = e
                return np.full(c.shape[0], self.dims[mu]) if kind == "x" else c[:, mu]
            per_block = np.zeros(c.shape[0])
            for t, below in enumerate(self.vertices):
                rows = np.prod([size(e) for e in below], axis=0)
                per_block += rows * (c[:, t] if t < b else 1)
            per_block = np.where(c.min(axis=1) > 0, per_block, 0)
            total += float(per_block.sum()) * g.real_params_per_entry
        return total

    @property
    def ranks_K(self) -> tuple:
        """Largest kept rank per vertex v != v_0 (in the order of T) over all blocks."""
        b = len(self.vertices) - 1
        return tuple(int(max(c[:, t].max() for c in self.counts)) for t in range(b))

    @property
    def error_bound2(self) -> float:
        return float(sum(self.tails))


def _check_commutative(K: Algebra, groups) -> None:
    if not K.is_commutative or any(g.m != 1 for g in groups):
        raise NoncommutativeTTNError(
            f"tree tensor networks over K are implemented only for commutative K; {K.name} "
            f"is noncommutative.  Use tt_svd_K for the TT path, which is valid for every K."
        )


def _rank_list(ranks, b):
    if ranks is None:
        return [None] * b
    if np.isscalar(ranks):
        return [int(ranks)] * b
    ranks = [int(r) for r in ranks]
    if len(ranks) != b:
        raise ValueError(f"this tree has {b} internal edges, got {len(ranks)} ranks")
    return ranks


def ttn_svd_K(X: np.ndarray, K: Algebra, tree, ranks=None, eps=None, adaptive=False,
              alg_axes=None) -> TTNK:
    """TTN-SVD of a tensor over a commutative K on a rooted tree (Algorithm 6).

    Parameters
    ----------
    X : array
        The tensor in coefficient form: (n_1, ..., n_N, *K.shape), or any layout with
        ``alg_axes`` naming the algebra axes.
    tree : nested tuples
        The rooted tree (see the module docstring); its ints are the modes 0..N-1.
    ranks : int or sequence of b ints, optional
        Uniform K-ranks r_v of the internal edges, for the vertices v != v_0 in the order
        of T (b = #V(T) - 1).
    eps : float, optional
        Relative tolerance: every step may discard eps^2 ||X||^2 / b.
    adaptive : bool
        With eps, choose the ranks separately in every block.
    """
    X = np.asarray(X)
    orig_ndim = X.ndim
    if alg_axes is not None:
        X = move_alg_axes_last(X, alg_axes)
    N = X.ndim - K.n_axes
    vertices, leaves = _parse_tree(tree)
    if sorted(leaves) != list(range(N)):
        raise ValueError(f"the external edges of the tree must be the modes 0..{N - 1}, "
                         f"each once; got {leaves}")
    b = len(vertices) - 1
    if b < 1:
        raise ValueError("a tree tensor network needs at least one internal edge")
    if ranks is not None and eps is not None:
        raise ValueError("give ranks or eps, not both")
    rl = _rank_list(ranks, b)
    # external edges by position in the drawing order
    order = tuple(leaves)
    pos = {mu: p for p, mu in enumerate(order)}
    vertices = [[(kind, pos[mu] if kind == "x" else mu) for kind, mu in below] for below in vertices]
    X = np.transpose(X, list(order) + list(range(N, X.ndim)))
    dims = X.shape[:N]
    groups = K.forward(X)
    _check_commutative(K, groups)
    total = float(K.norm2(X).sum())
    delta2 = None if eps is None else float(eps) ** 2 * total / b

    Ys = [np.moveaxis(g.data[..., 0, 0], N, 0) for g in groups]          # (B, n_1, ..., n_N)
    edges = [("x", p) for p in range(N)]
    cores = [[] for _ in groups]
    counts = [np.zeros((g.n_blocks, b), dtype=int) for g in groups]
    tails = []
    for t, below in enumerate(vertices[:-1]):
        p0 = edges.index(below[0])
        assert edges[p0:p0 + len(below)] == below, "edges below a vertex must be contiguous"
        mats, Us, spec = [], [], []
        for g, Y in zip(groups, Ys):
            M = _unfold(Y, p0, len(below))                               # cut the edge v - parent
            U, S = left_singular(M, g)
            mats.append(M)
            Us.append(U)
            spec.append(S)
        c = select_counts(spec, groups, rank=rl[t], delta2=delta2, adaptive=adaptive)
        tails.append(tail_energy(spec, groups, c))
        for gi, g in enumerate(groups):
            C = masked(Us[gi], c[gi], g)
            other = _others(Ys[gi], p0, len(below))
            Ys[gi] = _fold(np.conj(np.swapaxes(C, -1, -2)) @ mats[gi], (C.shape[-1],), other, p0)
            cores[gi].append(C)
            counts[gi][:, t] = c[gi]
        edges[p0:p0 + len(below)] = [("v", t)]
    assert edges == vertices[-1]
    return TTNK(K, tree, dims, order, vertices, groups, cores, Ys, counts, tails, total,
                alg_axes, orig_ndim)
