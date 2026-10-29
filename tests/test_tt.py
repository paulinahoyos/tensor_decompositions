"""Tests of the TT-SVD over K, for commutative and noncommutative K.

  * full ranks reproduce the tensor, in any mode order;
  * ||X - X_hat||^2 = sum_k tau_k (the Pythagorean identity of the nested projections);
  * tau_k <= the best rank-r_k error^2 of the k-th unfolding over K (Eckart-Young,
    Theorem 4.2), hence the sqrt(N - 1) quasi-optimality;
  * tolerance truncations (uniform and adaptive) satisfy ||X - X_hat|| <= eps ||X||;
  * uniform storage is (sum_k r_{k-1} n_k r_k) dim_R K;
  * K = R agrees with a direct numpy TT-SVD;
  * mode order and alg_axes are handled.
The quaternion structure of the cores is checked implicitly: an unstructured core would
not map back to K, and the reconstruction error would not equal the sum of the tails.
"""
import numpy as np
import pytest

from staralg.algebras.group import cyclic_table
from staralg import (
    DirectSum,
    Cl03,
    Cl04,
    Cl10,
    Cl30,
    MatrixAlgebra,
    Quaternions,
    TensorProduct,
    Trivial,
    TubeDCT,
    TubeDFT,
    AbelianGroupAlgebra,
    GroupAlgebra,
    WalshHadamard,
    svd_K,
    tt_svd_K,
)

ALGEBRAS = [
    AbelianGroupAlgebra(4, 3),
    AbelianGroupAlgebra(4, 4),
    AbelianGroupAlgebra(2, 3, 2),
    AbelianGroupAlgebra(3, 4, field="C"),
    GroupAlgebra.dihedral(4),
    GroupAlgebra.dihedral(5),
    GroupAlgebra.quaternion(),
    GroupAlgebra.symmetric(3),
    GroupAlgebra(cyclic_table(5), group_name="Z_5 from its table"),
    GroupAlgebra.product(GroupAlgebra.dihedral(3), TubeDFT(4)),
    GroupAlgebra.product(GroupAlgebra.quaternion(), TubeDFT(3)),
    GroupAlgebra.product(GroupAlgebra.quaternion(), GroupAlgebra.dihedral(3)),
    GroupAlgebra.dihedral(4, field="C"),
    GroupAlgebra.quaternion(field="C"),
    TensorProduct(GroupAlgebra.dihedral(3), TubeDCT(2)),
    Trivial("R"),
    Trivial("C"),
    TubeDFT(4),
    TubeDFT(5, "C"),
    TubeDCT(3),
    WalshHadamard(2),
    Cl10(),
    Quaternions(),
    Cl03(),
    Cl30(),
    Cl04(),
    MatrixAlgebra(2, "R"),
    MatrixAlgebra(2, "C"),
    MatrixAlgebra(2, "H"),
    TensorProduct(TubeDFT(4), TubeDCT(2)),
    TensorProduct(Cl03(), TubeDCT(2)),
    DirectSum(MatrixAlgebra(2, "C"), MatrixAlgebra(2, "C")),
    DirectSum(Trivial("C"), Trivial("C")),
    DirectSum(MatrixAlgebra(2, "R"), Trivial("R"), MatrixAlgebra(3, "R")),
]
DIMS = [(3, 4, 5), (2, 3, 4, 3), (4, 2, 3, 2, 2)]
TOL = dict(rtol=1e-8, atol=1e-8)


@pytest.fixture(params=ALGEBRAS, ids=lambda K: K.name)
def K(request):
    return request.param


@pytest.fixture(params=DIMS, ids=lambda d: "x".join(map(str, d)))
def dims(request):
    return request.param


def _err2(K, X, Y):
    return float(K.norm2(X - Y).sum())


def _rank_grid(N):
    yield tuple([1] * (N - 1))
    yield tuple([2] * (N - 1))
    yield tuple(1 + (k % 3) for k in range(N - 1))


def _unfolding(X, K, k):
    """The k-th unfolding over K: rows (i_1..i_k), columns (i_{k+1}..i_N)."""
    N = X.ndim - K.n_axes
    dims = X.shape[:N]
    return X.reshape((int(np.prod(dims[:k])), int(np.prod(dims[k:]))) + tuple(K.shape))


def test_full_rank_is_exact(K, dims):
    rng = np.random.default_rng(0)
    X = K.random(rng, dims)
    N = len(dims)
    big = int(np.prod(dims))
    for order in (tuple(range(N)), tuple(reversed(range(N)))):
        np.testing.assert_allclose(tt_svd_K(X, K, ranks=big, order=order).reconstruct(), X, **TOL)
    np.testing.assert_allclose(tt_svd_K(X, K).reconstruct(), X, **TOL)


def test_pythagorean_identity(K, dims):
    rng = np.random.default_rng(1)
    X = K.random(rng, dims)
    for r in _rank_grid(len(dims)):
        tt = tt_svd_K(X, K, ranks=r)
        np.testing.assert_allclose(_err2(K, X, tt.reconstruct()), sum(tt.tails), rtol=1e-8, atol=1e-10)


def test_step_tails_bounded_by_unfolding_svd(K, dims):
    """tau_k <= ||X_<k> - best rank-r_k approximation||^2 over K, for every step."""
    rng = np.random.default_rng(2)
    X = K.random(rng, dims)
    N = len(dims)
    for r in _rank_grid(N):
        tt = tt_svd_K(X, K, ranks=r)
        for k in range(N - 1):
            best = svd_K(_unfolding(X, K, k + 1), K).tail_energy(rank=r[k])
            assert tt.tails[k] <= best * (1 + 1e-9) + 1e-10


@pytest.mark.parametrize("adaptive", [False, True])
def test_tolerance(K, dims, adaptive):
    rng = np.random.default_rng(3)
    X = K.random(rng, dims)
    norm2 = float(K.norm2(X).sum())
    for eps in (0.05, 0.3, 0.6, 0.9):
        tt = tt_svd_K(X, K, eps=eps, adaptive=adaptive)
        assert _err2(K, X, tt.reconstruct()) <= eps ** 2 * norm2 * (1 + 1e-9)


def test_uniform_storage(K, dims):
    rng = np.random.default_rng(4)
    X = K.random(rng, dims)
    N = len(dims)
    for r in _rank_grid(N):
        tt = tt_svd_K(X, K, ranks=r)
        # ranks are capped by the unfolding sizes over K
        caps = [min(int(np.prod(dims[: k + 1])), int(np.prod(dims[k + 1:]))) for k in range(N - 1)]
        rr = [1] + [min(a, b) for a, b in zip(r, caps)] + [1]
        expected = sum(rr[k] * dims[k] * rr[k + 1] for k in range(N)) * K.real_dim
        assert tt.n_params() == expected
        assert tt.ranks_K == tuple(rr[1:-1])


def test_real_tt_matches_numpy():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((4, 5, 3, 6))
    ranks = (3, 4, 2)
    # reference TT-SVD (Oseledets 2011) with fixed ranks
    W, r_prev, cores = X.copy(), 1, []
    for k, r in enumerate(ranks):
        M = W.reshape(r_prev * X.shape[k], -1)
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        cores.append(U[:, :r].reshape(r_prev, X.shape[k], r))
        W = s[:r, None] * Vh[:r]
        r_prev = r
    cores.append(W.reshape(r_prev, X.shape[-1], 1))
    ref = cores[0]
    for C in cores[1:]:
        ref = np.tensordot(ref, C, axes=(-1, 0))
    ref = ref.reshape(X.shape)
    np.testing.assert_allclose(tt_svd_K(X, Trivial("R"), ranks=ranks).reconstruct(), ref, **TOL)


def test_order_argument(K):
    rng = np.random.default_rng(6)
    X = K.random(rng, (3, 4, 2))
    order = (2, 0, 1)
    Xp = np.transpose(X, list(order) + list(range(3, X.ndim)))
    a = tt_svd_K(X, K, ranks=(2, 2), order=order).reconstruct()
    b = tt_svd_K(Xp, K, ranks=(2, 2)).reconstruct()
    np.testing.assert_allclose(np.transpose(a, list(order) + list(range(3, X.ndim))), b, **TOL)


def test_alg_axes():
    rng = np.random.default_rng(7)
    K = Cl03()
    X = K.random(rng, (3, 4, 2))                       # (3, 4, 2, 8)
    Xf = np.moveaxis(X, -1, 0)                         # algebra axis first
    a = tt_svd_K(Xf, K, ranks=2, alg_axes=0).reconstruct()
    b = tt_svd_K(X, K, ranks=2).reconstruct()
    np.testing.assert_allclose(np.moveaxis(a, 0, -1), b, **TOL)


def test_wide_unfoldings_use_gram_path_correctly():
    """A first mode much smaller than the rest takes the Gram path; results must agree."""
    rng = np.random.default_rng(8)
    for K in (Cl03(), TubeDFT(4), Trivial("C"), MatrixAlgebra(2, "H")):
        X = K.random(rng, (2, 6, 5))
        tt = tt_svd_K(X, K, ranks=(1, 2))
        np.testing.assert_allclose(_err2(K, X, tt.reconstruct()), sum(tt.tails), rtol=1e-7, atol=1e-9)
        best = svd_K(_unfolding(X, K, 1), K).tail_energy(rank=1)
        np.testing.assert_allclose(tt.tails[0], best, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("K", [Cl03(), Cl04(), TubeDFT(4)], ids=lambda K: K.name)
def test_exact_low_rank_tt_is_recovered(K):
    """A tensor train of K-ranks (2, 3) is recovered exactly with those ranks."""
    rng = np.random.default_rng(9)
    from staralg import matmul_K

    G1 = K.random(rng, (4, 2))
    G2 = K.random(rng, (2 * 5, 3))
    G3 = K.random(rng, (3, 6))
    A = matmul_K(G1, G2.reshape((2, 5 * 3) + K.shape), K).reshape((4 * 5, 3) + K.shape)
    X = matmul_K(A, G3, K).reshape((4, 5, 6) + K.shape)
    tt = tt_svd_K(X, K, ranks=(2, 3))
    np.testing.assert_allclose(tt.reconstruct(), X, rtol=1e-8, atol=1e-8)
    tt = tt_svd_K(X, K, eps=1e-6, adaptive=True)
    assert tt.ranks_K == (2, 3)
