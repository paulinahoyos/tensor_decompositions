"""Tests of the SVD over K (Theorems 3.4 and 4.2, Proposition 3.13, Algorithms 1-2).

For each algebra and a random m x n matrix A over K:
  * A = U Sigma V^H with U^H U = V^H V = I over K (Theorem 3.4 / Algorithm 1);
  * Sigma is diagonal with Sigma_ii in K_+, and the SVD is ordered (Definition 3.8, Eq. 12);
  * ||A - A_k||^2 equals the weighted tail of discarded singular values;
  * Eckart-Young: no matrix of K-rank <= k is closer to A than A_k (Theorem 4.2);
  * rank_K(PQ) = r for random P (m x r), Q (r x n) (Proposition 3.13);
  * U_k U_k^H A = A_k with U_k from llsv_K (Eq. 24);
  * uniform storage is k (m + n) dim_R K;
  * adaptive truncations meet their energy and budget targets.
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
    conj_transpose_K,
    eye_K,
    llsv_K,
    matmul_K,
    rank_K,
    svd_K,
)

ALGEBRAS = [
    Trivial("R"),
    Trivial("C"),
    TubeDFT(4),
    TubeDFT(5),
    TubeDFT(4, "C"),
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
    TubeDCT(3),
    WalshHadamard(4),
    Cl10(),
    Quaternions(),
    Cl03(),
    Cl30(),
    Cl04(),
    MatrixAlgebra(2, "H"),
    MatrixAlgebra(2, "R"),
    MatrixAlgebra(2, "C"),
    TensorProduct(TubeDFT(4), TubeDCT(3)),
    TensorProduct(Cl03(), TubeDCT(2)),
    DirectSum(MatrixAlgebra(2, "C"), MatrixAlgebra(2, "C")),
    DirectSum(Trivial("C"), Trivial("C")),
    DirectSum(MatrixAlgebra(2, "R"), Trivial("R"), MatrixAlgebra(3, "R")),
]
SHAPES = [(5, 3), (3, 5), (4, 4)]
TOL = dict(rtol=1e-8, atol=1e-8)


@pytest.fixture(params=ALGEBRAS, ids=lambda K: K.name)
def K(request):
    return request.param


@pytest.fixture(params=SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def shape(request):
    return request.param


def _rand(K, rng, m, n):
    return K.random(rng, (m, n))


def _norm(K, X):
    return float(np.sqrt(K.norm2(X).sum()))


def test_full_reconstruction(K, shape):
    rng = np.random.default_rng(0)
    A = _rand(K, rng, *shape)
    np.testing.assert_allclose(svd_K(A, K).reconstruct(), A, **TOL)


def test_factors_are_an_svd(K, shape):
    rng = np.random.default_rng(1)
    m, n = shape
    A = _rand(K, rng, m, n)
    U, S, V = svd_K(A, K).factors()
    r = min(m, n)
    np.testing.assert_allclose(matmul_K(matmul_K(U, S, K), conj_transpose_K(V, K), K), A, **TOL)
    np.testing.assert_allclose(matmul_K(conj_transpose_K(U, K), U, K), eye_K(r, K), **TOL)
    np.testing.assert_allclose(matmul_K(conj_transpose_K(V, K), V, K), eye_K(r, K), **TOL)
    # Sigma is diagonal
    off = S.copy()
    for i in range(r):
        off[i, i] = 0
    np.testing.assert_allclose(off, 0, atol=1e-8)


def test_singular_values_positive_and_ordered(K, shape):
    """Sigma_ii in K_+ (Hermitian PSD blocks) and Eq. (12): sigma_min(Sigma_ii) >= sigma_max(Sigma_jj)."""
    rng = np.random.default_rng(2)
    A = _rand(K, rng, *shape)
    U, S, V = svd_K(A, K).factors()
    r = min(shape)
    blocks = [K.forward(S[i, i]) for i in range(r)]
    for i in range(r):
        for g in blocks[i]:
            M = g.data
            np.testing.assert_allclose(M, np.conj(np.swapaxes(M, -1, -2)), **TOL)
            assert np.linalg.eigvalsh(M).min() > -1e-9
    for i in range(r - 1):
        for gi, gj in zip(blocks[i], blocks[i + 1]):
            lo = np.linalg.eigvalsh(gi.data).min(axis=-1)
            hi = np.linalg.eigvalsh(gj.data).max(axis=-1)
            assert np.all(lo >= hi - 1e-9)


def test_tail_energy_equals_error(K, shape):
    rng = np.random.default_rng(3)
    A = _rand(K, rng, *shape)
    svd = svd_K(A, K)
    for k in range(1, min(shape) + 1):
        err2 = _norm(K, A - svd.reconstruct(rank=k)) ** 2
        np.testing.assert_allclose(err2, svd.tail_energy(rank=k), rtol=1e-8, atol=1e-10)
    for gamma in (0.5, 0.9, 0.99):
        err2 = _norm(K, A - svd.reconstruct(energy=gamma)) ** 2
        np.testing.assert_allclose(err2, svd.tail_energy(energy=gamma), rtol=1e-8, atol=1e-10)
        assert err2 <= (1 - gamma) * svd.total_energy * (1 + 1e-9) + 1e-12


def test_eckart_young(K, shape):
    """A_k beats random K-rank-k competitors and small perturbations of A_k itself."""
    rng = np.random.default_rng(4)
    m, n = shape
    A = _rand(K, rng, m, n)
    svd = svd_K(A, K)
    for k in range(1, min(m, n)):
        best = _norm(K, A - svd.reconstruct(rank=k))
        U, S, V = svd.factors(k)
        SVh = matmul_K(S, conj_transpose_K(V, K), K)
        for _ in range(20):
            P, Q = _rand(K, rng, m, k), _rand(K, rng, k, n)
            assert _norm(K, A - matmul_K(P, Q, K)) >= best - 1e-9
            eps = 1e-3
            P2 = U + eps * _rand(K, rng, m, k)
            Q2 = SVh + eps * _rand(K, rng, k, n)
            assert _norm(K, A - matmul_K(P2, Q2, K)) >= best - 1e-9


def test_rank_formula(K):
    rng = np.random.default_rng(5)
    m, n = 6, 5
    for r in range(1, 5):
        A = matmul_K(_rand(K, rng, m, r), _rand(K, rng, r, n), K)
        assert rank_K(A, K) == r
        np.testing.assert_allclose(svd_K(A, K).reconstruct(rank=r), A, **TOL)


def test_llsv_projection(K, shape):
    rng = np.random.default_rng(6)
    m, n = shape
    A = _rand(K, rng, m, n)
    svd = svd_K(A, K)
    for k in range(1, min(m, n) + 1):
        Uk = llsv_K(A, K, k)
        PA = matmul_K(Uk, matmul_K(conj_transpose_K(Uk, K), A, K), K)
        np.testing.assert_allclose(PA, svd.reconstruct(rank=k), **TOL)


def test_llsv_wide_matrix_uses_gram_correctly(K):
    rng = np.random.default_rng(7)
    A = _rand(K, rng, 3, 12)  # cols > 2 rows: eigh path for non-quaternion blocks
    Uk = llsv_K(A, K, 2)
    PA = matmul_K(Uk, matmul_K(conj_transpose_K(Uk, K), A, K), K)
    np.testing.assert_allclose(PA, svd_K(A, K).reconstruct(rank=2), **TOL)


def test_uniform_storage(K, shape):
    rng = np.random.default_rng(8)
    m, n = shape
    svd = svd_K(_rand(K, rng, m, n), K)
    for k in range(1, min(m, n) + 1):
        assert svd.n_params(rank=k) == k * (m + n) * K.real_dim


def test_budget_truncation(K):
    rng = np.random.default_rng(9)
    m, n = 6, 5
    A = _rand(K, rng, m, n)
    svd = svd_K(A, K)
    for k in (1, 2, 3):
        budget = svd.n_params(rank=k)
        assert svd.n_params(budget=budget) <= budget
        # adaptive truncation within the same budget is at least as good when all blocks
        # have equal cost per singular value (always true for one-group algebras)
        if len(svd.templates) == 1:
            assert svd.tail_energy(budget=budget) <= svd.tail_energy(rank=k) + 1e-9


def test_trivial_algebra_matches_numpy():
    rng = np.random.default_rng(10)
    A = rng.standard_normal((7, 5))
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    svd = svd_K(A, Trivial("R"))
    for k in range(1, 6):
        Ak = (U[:, :k] * s[:k]) @ Vh[:k]
        np.testing.assert_allclose(svd.reconstruct(rank=k), Ak, **TOL)


def test_alg_axes_argument():
    rng = np.random.default_rng(11)
    K = TubeDFT(4)
    A = K.random(rng, (5, 3))                      # (5, 3, 4)
    A_first = np.moveaxis(A, -1, 0)                # algebra axis first: (4, 5, 3)
    out = svd_K(A_first, K, alg_axes=0).reconstruct(rank=2)
    np.testing.assert_allclose(np.moveaxis(out, 0, -1), svd_K(A, K).reconstruct(rank=2), **TOL)


@pytest.mark.parametrize("K", [Cl03(), Cl04(), MatrixAlgebra(2, "H")], ids=lambda K: K.name)
def test_quaternion_singular_values_come_in_pairs(K):
    rng = np.random.default_rng(12)
    svd = svd_K(K.random(rng, (5, 4)), K)
    (S,) = svd.singular_values()
    np.testing.assert_allclose(S[:, 0::2], S[:, 1::2], rtol=1e-10)


@pytest.mark.parametrize("K", [Cl03(), Quaternions(), Cl04(), MatrixAlgebra(2, "H")], ids=lambda K: K.name)
def test_quaternion_degenerate_and_rank_deficient(K):
    """Repeated singular values (clusters larger than a pair) and null spaces."""
    rng = np.random.default_rng(13)
    n = 4
    # an isometry over K: all singular values equal
    U, _, _ = svd_K(K.random(rng, (n, n)), K).factors()
    # a rank-2 matrix with a repeated singular value: 3 (U_1 U_1^H + U_2 U_2^H)
    P = matmul_K(U[:, :2], conj_transpose_K(U[:, :2], K), K)
    for A in (U, 3.0 * P, np.zeros_like(U)):
        svd = svd_K(A, K)
        np.testing.assert_allclose(svd.reconstruct(), A, **TOL)
        Uf, S, Vf = svd.factors()
        np.testing.assert_allclose(matmul_K(matmul_K(Uf, S, K), conj_transpose_K(Vf, K), K), A, **TOL)
        np.testing.assert_allclose(matmul_K(conj_transpose_K(Uf, K), Uf, K), eye_K(n, K), **TOL)
        np.testing.assert_allclose(matmul_K(conj_transpose_K(Vf, K), Vf, K), eye_K(n, K), **TOL)
