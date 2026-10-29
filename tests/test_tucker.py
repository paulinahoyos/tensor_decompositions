"""Tests of the Tucker decompositions over a commutative K (HOSVD and ST-HOSVD).

  * full ranks reproduce the tensor;
  * HOSVD: tau_k <= ||X - X_hat||^2 <= sum_k tau_k;
  * ST-HOSVD: ||X - X_hat||^2 = sum_k tau_k (Pythagorean identity of nested projections);
  * both are quasi-optimal: ||X - X_hat||^2 <= N * (best error^2), checked against the
    lower bound max_k tau_k^HOSVD <= best error^2;
  * tolerance truncations (uniform and adaptive) satisfy ||X - X_hat|| <= eps ||X||;
  * uniform storage is (prod_k r_k + sum_k n_k r_k) dim_R K;
  * K = R agrees with a direct numpy HOSVD;
  * noncommutative K raises NoncommutativeTuckerError;
  * an order-3 TT with ranks (r1, r2) equals the ST-HOSVD with ranks (r1, n2, r2) and
    mode order (0, 2, 1).
"""
import numpy as np
import pytest

from staralg.algebras.group import abelian_table, cyclic_table
from staralg import (
    DirectSum,
    Cl03,
    Cl04,
    Cl10,
    Cl30,
    MatrixAlgebra,
    NoncommutativeTuckerError,
    Quaternions,
    TensorProduct,
    Trivial,
    TubeDCT,
    TubeDFT,
    AbelianGroupAlgebra,
    GroupAlgebra,
    WalshHadamard,
    hosvd_K,
    sthosvd_K,
    tt_svd_K,
)

COMMUTATIVE = [
    AbelianGroupAlgebra(4, 3),
    AbelianGroupAlgebra(4, 4),
    AbelianGroupAlgebra(2, 3, 2),
    AbelianGroupAlgebra(3, 4, field="C"),
    GroupAlgebra(cyclic_table(5), group_name="Z_5 from its table"),
    GroupAlgebra(abelian_table((2, 3)), shape=(2, 3), group_name="Z_2 x Z_3 from its table"),
    Trivial("R"),
    Trivial("C"),
    TubeDFT(4),
    TubeDFT(5),
    TubeDFT(4, "C"),
    TubeDCT(3),
    WalshHadamard(4),
    Cl10(),
    TensorProduct(TubeDFT(4), TubeDCT(3)),
    TensorProduct(TubeDFT(3, "C"), TubeDCT(2)),
    DirectSum(Trivial("C"), Trivial("C")),
]
NONCOMMUTATIVE = [GroupAlgebra.dihedral(4), GroupAlgebra.quaternion(), DirectSum(MatrixAlgebra(2, "C"), MatrixAlgebra(2, "C")), Cl03(), Quaternions(), Cl30(), Cl04(), MatrixAlgebra(2, "R"),
                  TensorProduct(Cl03(), TubeDCT(2))]
DIMS = [(4, 3, 5), (3, 4, 2, 3)]
TOL = dict(rtol=1e-8, atol=1e-8)


@pytest.fixture(params=COMMUTATIVE, ids=lambda K: K.name)
def K(request):
    return request.param


@pytest.fixture(params=DIMS, ids=lambda d: "x".join(map(str, d)))
def dims(request):
    return request.param


def _err2(K, X, Y):
    return float(K.norm2(X - Y).sum())


def _rank_grid(dims):
    N = len(dims)
    yield tuple([1] * N)
    yield tuple(max(1, n - 1) for n in dims)
    yield tuple(min(2, n) for n in dims)
    yield tuple(dims)


def test_full_rank_is_exact(K, dims):
    rng = np.random.default_rng(0)
    X = K.random(rng, dims)
    np.testing.assert_allclose(hosvd_K(X, K).truncate(ranks=dims).reconstruct(), X, **TOL)
    np.testing.assert_allclose(sthosvd_K(X, K, ranks=dims).reconstruct(), X, **TOL)


def test_hosvd_error_bounds(K, dims):
    rng = np.random.default_rng(1)
    X = K.random(rng, dims)
    h = hosvd_K(X, K)
    for r in _rank_grid(dims):
        T = h.truncate(ranks=r)
        e2 = _err2(K, X, T.reconstruct())
        assert e2 <= sum(T.tails) * (1 + 1e-9) + 1e-10
        assert e2 >= max(T.tails) * (1 - 1e-9) - 1e-10


def test_sthosvd_pythagorean(K, dims):
    rng = np.random.default_rng(2)
    X = K.random(rng, dims)
    N = len(dims)
    for r in _rank_grid(dims):
        for order in (tuple(range(N)), tuple(reversed(range(N)))):
            T = sthosvd_K(X, K, ranks=r, order=order)
            np.testing.assert_allclose(_err2(K, X, T.reconstruct()), sum(T.tails), rtol=1e-8, atol=1e-10)


def test_quasi_optimality(K, dims):
    """Both errors are at most N times a lower bound on the best error (max_k HOSVD tail)."""
    rng = np.random.default_rng(3)
    X = K.random(rng, dims)
    h = hosvd_K(X, K)
    N = len(dims)
    for r in _rank_grid(dims):
        lower = max(h.truncate(ranks=r).tails)          # <= best error^2 (Eckart-Young per mode)
        for T in (h.truncate(ranks=r), sthosvd_K(X, K, ranks=r)):
            assert _err2(K, X, T.reconstruct()) <= N * lower * (1 + 1e-9) + 1e-10


@pytest.mark.parametrize("adaptive", [False, True])
def test_tolerance(K, dims, adaptive):
    rng = np.random.default_rng(4)
    X = K.random(rng, dims)
    norm2 = float(K.norm2(X).sum())
    for eps in (0.05, 0.3, 0.6, 0.9):
        for T in (hosvd_K(X, K).truncate(eps=eps, adaptive=adaptive),
                  sthosvd_K(X, K, eps=eps, adaptive=adaptive)):
            assert _err2(K, X, T.reconstruct()) <= eps ** 2 * norm2 * (1 + 1e-9)


def test_adaptive_never_stores_more_than_needed_for_uniform(K, dims):
    """For the same tolerance, adaptive HOSVD keeps at most the uniform ranks in every block."""
    rng = np.random.default_rng(5)
    X = K.random(rng, dims)
    h = hosvd_K(X, K)
    for eps in (0.2, 0.5):
        Tu, Ta = h.truncate(eps=eps), h.truncate(eps=eps, adaptive=True)
        assert Ta.n_params() <= Tu.n_params() + 1e-9


def test_uniform_storage(K, dims):
    rng = np.random.default_rng(6)
    X = K.random(rng, dims)
    h = hosvd_K(X, K)
    for r in _rank_grid(dims):
        expected = (np.prod(r) + sum(n * k for n, k in zip(dims, r))) * K.real_dim
        assert h.truncate(ranks=r).n_params() == expected
        assert sthosvd_K(X, K, ranks=r).n_params() == expected


def test_real_hosvd_matches_numpy():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((5, 4, 6))
    r = (2, 3, 2)
    Us = []
    for k in range(3):
        Mk = np.moveaxis(X, k, 0).reshape(X.shape[k], -1)
        Us.append(np.linalg.svd(Mk, full_matrices=False)[0][:, : r[k]])
    G = np.einsum("abc,ai,bj,ck->ijk", X, *Us)
    ref = np.einsum("ijk,ai,bj,ck->abc", G, *Us)
    np.testing.assert_allclose(hosvd_K(X, Trivial("R")).truncate(ranks=r).reconstruct(), ref, **TOL)


def test_alg_axes():
    rng = np.random.default_rng(8)
    K = TubeDFT(4)
    X = K.random(rng, (3, 5, 2))                       # (3, 5, 2, 4)
    Xf = np.moveaxis(X, -1, 1)                         # algebra axis second: (3, 4, 5, 2)
    a = sthosvd_K(Xf, K, ranks=(2, 2, 1), alg_axes=1).reconstruct()
    b = sthosvd_K(X, K, ranks=(2, 2, 1)).reconstruct()
    np.testing.assert_allclose(np.moveaxis(a, 1, -1), b, **TOL)


@pytest.mark.parametrize("Knc", NONCOMMUTATIVE, ids=lambda K: K.name)
def test_noncommutative_raises(Knc):
    rng = np.random.default_rng(9)
    X = Knc.random(rng, (3, 3, 2))
    with pytest.raises(NoncommutativeTuckerError):
        hosvd_K(X, Knc)
    with pytest.raises(NoncommutativeTuckerError):
        sthosvd_K(X, Knc, ranks=1)


def test_order3_tt_equals_sthosvd(K):
    rng = np.random.default_rng(10)
    dims = (4, 3, 5)
    X = K.random(rng, dims)
    for r1, r2 in [(1, 1), (2, 3), (3, 2), (4, 5)]:
        tt = tt_svd_K(X, K, ranks=(r1, r2)).reconstruct()
        st = sthosvd_K(X, K, ranks=(r1, dims[1], r2), order=(0, 2, 1)).reconstruct()
        np.testing.assert_allclose(tt, st, **TOL)
