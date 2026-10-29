"""Tests of the TTN-SVD over a commutative K (Algorithm 6).

  * full ranks reproduce the tensor, on every test tree;
  * ||X - X_hat||^2 = sum_v tau_v (the Pythagorean identity of the nested projections);
  * tau_v <= the best rank-r_v error^2 of the unfolding of X along the edge of v, hence the
    sqrt(#V(T) - 1) quasi-optimality of Theorem 5.12;
  * tolerance truncations (uniform and adaptive) satisfy ||X - X_hat|| <= eps ||X||;
  * on the TT path it agrees with tt_svd_K, on the Tucker star with sthosvd_K;
  * a root without external edges and two children of equal size: the second child
    discards nothing;
  * the coefficient-form cores reproduce the tensor through Eq. (33), evaluated with K.mul,
    and have orthonormal columns over K;
  * an exact low-rank TTN is recovered; uniform storage; leaf order and alg_axes;
  * a noncommutative K and malformed trees are refused.
"""
import itertools

import numpy as np
import pytest

from staralg.algebras.group import cyclic_table
from staralg import (
    AbelianGroupAlgebra,
    Cl10,
    DirectSum,
    GroupAlgebra,
    MatrixAlgebra,
    NoncommutativeTTNError,
    NoncommutativeTuckerError,
    Quaternions,
    TensorProduct,
    Trivial,
    TubeDCT,
    TubeDFT,
    WalshHadamard,
    conj_transpose_K,
    eye_K,
    ht_tree,
    matmul_K,
    sthosvd_K,
    svd_K,
    tt_svd_K,
    tt_tree,
    ttn_svd_K,
    tucker_tree,
)

ALGEBRAS = [
    Trivial("R"),
    Trivial("C"),
    TubeDFT(4),
    TubeDFT(5, "C"),
    TubeDCT(3),
    WalshHadamard(2),
    Cl10(),
    AbelianGroupAlgebra(2, 3),
    AbelianGroupAlgebra(3, 4, field="C"),
    GroupAlgebra(cyclic_table(5), group_name="Z_5 from its table"),
    TensorProduct(TubeDFT(4), TubeDCT(2)),
    DirectSum(Trivial("C"), Trivial("C")),
]
FIG3 = (((0,), (1, 2)), (3,), (4,))
# (tree, mode sizes)
TREES = [
    (FIG3, (2, 3, 2, 3, 2)),
    (tt_tree(4), (2, 3, 4, 3)),
    (tucker_tree(3), (3, 4, 5)),
    (ht_tree(4), (3, 2, 3, 2)),
    ((((0,), 1), (2, 3)), (2, 3, 2, 3)),     # a vertex with a child and an external edge
    (((0, 1), 2), (3, 2, 4)),                 # a root with an external edge
    (((0, 1),), (3, 4)),                      # a root with a single child
]
TOL = dict(rtol=1e-8, atol=1e-8)


@pytest.fixture(params=ALGEBRAS, ids=lambda K: K.name)
def K(request):
    return request.param


@pytest.fixture(params=TREES, ids=lambda td: str(td[0]).replace(" ", ""))
def tree_dims(request):
    return request.param


def _err2(K, X, Y):
    return float(K.norm2(X - Y).sum())


def _n_internal(tree):
    return sum(isinstance(item, tuple) for item in tree) + sum(
        _n_internal(item) for item in tree if isinstance(item, tuple))


def _rank_grid(b):
    yield tuple([1] * b)
    yield tuple([2] * b)
    yield tuple(1 + (k % 3) for k in range(b))


def _subtree_modes(ttn, t):
    """External positions below vertex t."""
    out = []
    for kind, mu in ttn.vertices[t]:
        out += [mu] if kind == "x" else _subtree_modes(ttn, mu)
    return out


def _eq33(K, ttn, cores):
    """Eq. (33) in coefficient form: sum over j of the product of the cores in the order of
    T, with K.mul, for all external indices at once."""
    N, b = len(ttn.dims), len(ttn.vertices) - 1
    out = np.zeros(ttn.dims + tuple(K.shape), dtype=np.result_type(*cores))
    for j in itertools.product(*[range(r) for r in ttn.ranks_K]):
        prod = None
        for t, below in enumerate(ttn.vertices):
            idx = tuple(slice(None) if kind == "x" else j[mu] for kind, mu in below)
            F = cores[t][idx + ((j[t],) if t < b else ())]
            ext = [mu for kind, mu in below if kind == "x"]
            shape = [ttn.dims[p] if p in ext else 1 for p in range(N)] + list(K.shape)
            F = np.broadcast_to(F.reshape(shape), out.shape)
            prod = F if prod is None else K.mul(prod, F)
        out = out + prod
    return out


def test_full_rank_is_exact(K, tree_dims):
    tree, dims = tree_dims
    X = K.random(np.random.default_rng(0), dims)
    np.testing.assert_allclose(ttn_svd_K(X, K, tree).reconstruct(), X, **TOL)
    np.testing.assert_allclose(ttn_svd_K(X, K, tree, ranks=int(np.prod(dims))).reconstruct(), X, **TOL)


def test_pythagorean_identity(K, tree_dims):
    tree, dims = tree_dims
    X = K.random(np.random.default_rng(1), dims)
    for r in _rank_grid(_n_internal(tree)):
        ttn = ttn_svd_K(X, K, tree, ranks=r)
        np.testing.assert_allclose(_err2(K, X, ttn.reconstruct()), sum(ttn.tails), rtol=1e-8, atol=1e-10)


def test_tails_bounded_by_unfolding_svd(K, tree_dims):
    """tau_v <= ||X_(v) - best rank-r_v approximation||^2 over K (Theorem 5.12)."""
    tree, dims = tree_dims
    X = K.random(np.random.default_rng(2), dims)
    for r in _rank_grid(_n_internal(tree)):
        ttn = ttn_svd_K(X, K, tree, ranks=r)
        for t in range(len(r)):
            rows = sorted(_subtree_modes(ttn, t))
            Xv = np.moveaxis(X, rows, list(range(len(rows))))
            M = Xv.reshape((int(np.prod([dims[p] for p in rows])), -1) + tuple(K.shape))
            best = svd_K(M, K).tail_energy(rank=r[t])
            assert ttn.tails[t] <= best * (1 + 1e-9) + 1e-10


@pytest.mark.parametrize("adaptive", [False, True])
def test_tolerance(K, tree_dims, adaptive):
    tree, dims = tree_dims
    X = K.random(np.random.default_rng(3), dims)
    norm2 = float(K.norm2(X).sum())
    for eps in (0.05, 0.3, 0.6, 0.9):
        ttn = ttn_svd_K(X, K, tree, eps=eps, adaptive=adaptive)
        assert _err2(K, X, ttn.reconstruct()) <= eps ** 2 * norm2 * (1 + 1e-9)


def test_tt_path_matches_tt_svd(K):
    dims = (2, 3, 4, 3)
    X = K.random(np.random.default_rng(4), dims)
    for kw in [dict(ranks=(1, 2, 3)), dict(ranks=2), dict(eps=0.4), dict(eps=0.4, adaptive=True)]:
        a = ttn_svd_K(X, K, tt_tree(4), **kw)
        b = tt_svd_K(X, K, **kw)
        np.testing.assert_allclose(a.reconstruct(), b.reconstruct(), **TOL)
        np.testing.assert_allclose(a.tails, b.tails, rtol=1e-8, atol=1e-10)
        assert a.n_params() == b.n_params()


def test_tucker_star_matches_sthosvd(K):
    dims = (3, 4, 5)
    X = K.random(np.random.default_rng(5), dims)
    for kw in [dict(ranks=(2, 3, 2)), dict(eps=0.4), dict(eps=0.4, adaptive=True)]:
        a = ttn_svd_K(X, K, tucker_tree(3), **kw)
        b = sthosvd_K(X, K, order=(0, 1, 2), **kw)
        np.testing.assert_allclose(a.reconstruct(), b.reconstruct(), **TOL)
        np.testing.assert_allclose(a.tails, b.tails, rtol=1e-8, atol=1e-10)
        assert a.n_params() == b.n_params()


def test_balanced_root_second_child_discards_nothing(K):
    """Root with no external edge and children u, w with r_u = r_w: tau_w = 0."""
    X = K.random(np.random.default_rng(6), (3, 2, 3, 2))
    ttn = ttn_svd_K(X, K, ht_tree(4), ranks=(2, 2, 3, 2, 2, 3))
    assert ttn.tails[-1] <= 1e-10 * ttn.total_energy
    assert ttn.tails[2] > 1e-6 * ttn.total_energy        # the first child does truncate


@pytest.mark.parametrize("adaptive", [False, True])
def test_cores_reproduce_eq33(K, adaptive):
    X = K.random(np.random.default_rng(7), (2, 3, 2, 3, 2))
    kw = dict(eps=0.3, adaptive=True) if adaptive else dict(ranks=(2, 2, 3, 2, 2))
    ttn = ttn_svd_K(X, K, FIG3, **kw)
    np.testing.assert_allclose(_eq33(K, ttn, ttn.cores_K()), ttn.reconstruct(), **TOL)


def test_cores_have_orthonormal_columns(K):
    X = K.random(np.random.default_rng(8), (2, 3, 2, 3, 2))
    ttn = ttn_svd_K(X, K, FIG3, ranks=(2, 2, 3, 2, 2))
    for r, C in zip(ttn.ranks_K, ttn.cores_K()):
        M = C.reshape((-1, r) + tuple(K.shape))
        np.testing.assert_allclose(matmul_K(conj_transpose_K(M, K), M, K), eye_K(r, K), **TOL)


def test_exact_low_rank_ttn_is_recovered(K):
    rng = np.random.default_rng(9)
    dims, ranks = (3, 4, 2, 4, 3), (2, 3, 4, 2, 2)
    ttn = ttn_svd_K(K.random(rng, dims), K, FIG3, ranks=ranks)
    cores = [K.random(rng, C.shape[:C.ndim - K.n_axes]) for C in ttn.cores_K()]
    X = _eq33(K, ttn, cores)
    np.testing.assert_allclose(ttn_svd_K(X, K, FIG3, ranks=ranks).reconstruct(), X, **TOL)
    assert ttn_svd_K(X, K, FIG3, eps=1e-6, adaptive=True).ranks_K == ranks


def test_uniform_storage(K):
    X = K.random(np.random.default_rng(10), (2, 3, 2, 3, 2))
    ttn = ttn_svd_K(X, K, FIG3, ranks=(2, 2, 3, 2, 2))
    # C_1: 2*2, C_2: 3*2*2, C_3: 2*2*3, C_4: 3*2, C_5: 2*2, root: 3*2*2
    assert ttn.n_params() == 50 * K.real_dim
    assert ttn.ranks_K == (2, 2, 3, 2, 2)


def test_leaf_order(K):
    X = K.random(np.random.default_rng(11), (2, 3, 4, 3))
    order = (2, 0, 3, 1)
    Xp = np.transpose(X, list(order) + list(range(4, X.ndim)))
    a = ttn_svd_K(X, K, ((2,), (0, (3,)), 1), ranks=2).reconstruct()
    b = ttn_svd_K(Xp, K, ((0,), (1, (2,)), 3), ranks=2).reconstruct()
    np.testing.assert_allclose(np.transpose(a, list(order) + list(range(4, X.ndim))), b, **TOL)


def test_alg_axes():
    K = TubeDFT(4)
    X = K.random(np.random.default_rng(12), (2, 3, 2, 3, 2))
    a = ttn_svd_K(np.moveaxis(X, -1, 0), K, FIG3, ranks=2, alg_axes=0).reconstruct()
    b = ttn_svd_K(X, K, FIG3, ranks=2).reconstruct()
    np.testing.assert_allclose(np.moveaxis(a, 0, -1), b, **TOL)


@pytest.mark.parametrize("K", [Quaternions(), MatrixAlgebra(2, "R"), GroupAlgebra.dihedral(3)],
                         ids=lambda K: K.name)
def test_noncommutative_is_refused(K):
    X = K.random(np.random.default_rng(13), (2, 3, 2))
    with pytest.raises(NoncommutativeTTNError):
        ttn_svd_K(X, K, tucker_tree(3))
    with pytest.raises(NoncommutativeTuckerError):
        ttn_svd_K(X, K, tt_tree(3))


def test_malformed_trees_are_refused():
    K = Trivial("R")
    X = K.random(np.random.default_rng(14), (2, 3, 2))
    for tree in [((0,), (1,)), ((0,), (1,), (1,)), ((0,), (1,), (2,), ()), (0, 1, 2), ((0,), (1,), (3,))]:
        with pytest.raises(ValueError):
            ttn_svd_K(X, K, tree)
    with pytest.raises(ValueError):
        ttn_svd_K(X, K, tucker_tree(3), ranks=(1, 2))
    with pytest.raises(ValueError):
        ttn_svd_K(X, K, tucker_tree(3), ranks=1, eps=0.1)
