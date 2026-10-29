"""Checks that every structure map is a *-isomorphism with the documented weights.

For random elements a, b, c of each algebra K:
  * inverse(forward(a)) == a
  * forward(ab) == forward(a) forward(b)            (blockwise matrix products)
  * forward(a^*) == forward(a)^H                    (blockwise conjugate transpose)
  * ||a||^2 == sum_rho c_rho ||pi_rho(a)||_F^2      (the weights)
  * 1 a == a == a 1 and forward(1) is the identity in every block
  * <ab, c> == <b, a^* c> and <ba, c> == <b, c a^*> (adjoint identities)
  * the real parameter count of the blocks equals dim_R K
"""
import numpy as np
import pytest

from staralg.algebras.group import cyclic_table
from staralg.algebras import (
    DirectSum,
    Cl03,
    Cl04,
    Cl10,
    Cl30,
    MatrixAlgebra,
    Quaternions,
    RandomOrthogonal,
    TensorProduct,
    Trivial,
    TubeDCT,
    TubeDFT,
    AbelianGroupAlgebra,
    GroupAlgebra,
    WalshHadamard,
)

ALGEBRAS = [
    Trivial("R"),
    Trivial("C"),
    TubeDFT(1),
    TubeDFT(2),
    TubeDFT(4),
    TubeDFT(5),
    TubeDFT(4, "C"),
    TubeDFT(5, "C"),
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
    GroupAlgebra.symmetric(4),
    TubeDCT(5),
    WalshHadamard(4),
    RandomOrthogonal(4, seed=1),
    Cl10(),
    Cl03(),
    Cl30(),
    Quaternions(),
    Cl04(),
    MatrixAlgebra(1, "H"),
    MatrixAlgebra(2, "H"),
    MatrixAlgebra(3, "R"),
    MatrixAlgebra(2, "C"),
    TensorProduct(Cl04(), TubeDCT(2)),
    TensorProduct(TubeDFT(4), TubeDCT(3)),
    TensorProduct(TubeDCT(3), TubeDFT(5)),
    TensorProduct(TubeDFT(4, "C"), TubeDCT(3)),
    TensorProduct(Cl03(), TubeDCT(3)),
    TensorProduct(TubeDCT(2), Cl10(), WalshHadamard(2)),
    TensorProduct(TubeDCT(3), WalshHadamard(2)),
    DirectSum(MatrixAlgebra(2, "C"), MatrixAlgebra(2, "C")),
    DirectSum(Trivial("C"), Trivial("C")),
    DirectSum(MatrixAlgebra(2, "R"), Trivial("R"), MatrixAlgebra(3, "R")),
]

BATCH = (3, 2)
TOL = dict(rtol=1e-10, atol=1e-10)


@pytest.fixture(params=ALGEBRAS, ids=lambda K: K.name)
def K(request):
    return request.param


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _matmul(g1, g2):
    return [a.data @ b.data for a, b in zip(g1, g2)]


def _same_blocks(blocks1, blocks2):
    assert len(blocks1) == len(blocks2)
    for x, y in zip(blocks1, blocks2):
        np.testing.assert_allclose(x, y, **TOL)


def test_roundtrip(K, rng):
    a = K.random(rng, BATCH)
    np.testing.assert_allclose(K.inverse(K.forward(a)), a, **TOL)


def test_multiplicative(K, rng):
    a, b = K.random(rng, BATCH), K.random(rng, BATCH)
    lhs = [g.data for g in K.forward(K.mul(a, b))]
    _same_blocks(lhs, _matmul(K.forward(a), K.forward(b)))


def test_star(K, rng):
    a = K.random(rng, BATCH)
    lhs = [g.data for g in K.forward(K.star(a))]
    rhs = [np.conj(np.swapaxes(g.data, -1, -2)) for g in K.forward(a)]
    _same_blocks(lhs, rhs)


def test_weights(K, rng):
    a = K.random(rng, BATCH)
    groups = K.forward(a)
    block_norms = sum(
        np.einsum("b,...b->...", g.weights, np.sum(np.abs(g.data) ** 2, axis=(-1, -2))) for g in groups
    )
    np.testing.assert_allclose(K.norm2(a), block_norms, **TOL)


def test_unit(K, rng):
    a = K.random(rng, BATCH)
    one = K.one()
    np.testing.assert_allclose(K.mul(one, a), a, **TOL)
    np.testing.assert_allclose(K.mul(a, one), a, **TOL)
    for g in K.forward(one):
        eye = np.broadcast_to(np.eye(g.m), g.data.shape)
        np.testing.assert_allclose(g.data, eye, **TOL)


def test_adjoint_identities(K, rng):
    a, b, c = (K.random(rng, BATCH) for _ in range(3))
    np.testing.assert_allclose(K.inner(K.mul(a, b), c), K.inner(b, K.mul(K.star(a), c)), **TOL)
    np.testing.assert_allclose(K.inner(K.mul(b, a), c), K.inner(b, K.mul(c, K.star(a))), **TOL)


def test_commutativity_flag(K, rng):
    a, b = K.random(rng, (5,)), K.random(rng, (5,))
    commutes = np.allclose(K.mul(a, b), K.mul(b, a))
    assert commutes == K.is_commutative


def test_parameter_count(K):
    zero = np.zeros(K.shape, dtype=float if K.field == "R" else complex)
    count = sum(g.n_blocks * g.m * g.m * g.real_params_per_entry for g in K.forward(zero))
    assert count == K.real_dim


def test_quaternion_blocks_have_quaternion_structure(K, rng):
    a = K.random(rng, BATCH)
    for g in K.forward(a):
        if g.division != "H":
            continue
        assert g.m % 2 == 0
        blocks = g.data.reshape(g.data.shape[:-2] + (g.m // 2, 2, g.m // 2, 2))
        alpha, beta = blocks[..., 0, :, 0], blocks[..., 0, :, 1]
        np.testing.assert_allclose(blocks[..., 1, :, 1], np.conj(alpha), **TOL)
        np.testing.assert_allclose(blocks[..., 1, :, 0], -np.conj(beta), **TOL)


# ---------------------------------------------------------------------------
# Specific structure maps
# ---------------------------------------------------------------------------
def _quaternion_vector_part(Q):
    """(b, c, d) of q = a + bi + cj + dk from its complex 2x2 realization."""
    return np.stack([Q[..., 0, 0].imag, Q[..., 0, 1].real, Q[..., 0, 1].imag], axis=-1)


def test_cl03_blocks_are_elsasser_fields(rng):
    K = Cl03()
    u = rng.standard_normal((4, 3))
    b = rng.standard_normal((4, 3))
    x = K.encode(vector=u, dual=b)
    np.testing.assert_allclose(K.decode(x)["vector"], u, **TOL)
    np.testing.assert_allclose(K.decode(x)["dual"], b, **TOL)
    (g,) = K.forward(x)
    plus, minus = g.data[..., 0, :, :], g.data[..., 1, :, :]
    np.testing.assert_allclose(_quaternion_vector_part(plus), -(u + b), **TOL)
    np.testing.assert_allclose(_quaternion_vector_part(minus), u - b, **TOL)
    # no scalar parts
    np.testing.assert_allclose(plus[..., 0, 0].real, 0, atol=1e-12)
    np.testing.assert_allclose(minus[..., 0, 0].real, 0, atol=1e-12)


def test_cl10_blocks_are_sum_and_difference(rng):
    K = Cl10()
    u, b = rng.standard_normal(6), rng.standard_normal(6)
    (g,) = K.forward(np.stack([u, b], axis=-1))
    np.testing.assert_allclose(g.data[..., 0, 0, 0], u + b, **TOL)
    np.testing.assert_allclose(g.data[..., 1, 0, 0], u - b, **TOL)


def test_cl03_pseudoscalar_is_central_and_selfadjoint():
    K = Cl03()
    I = K.pseudoscalar()
    np.testing.assert_allclose(K.star(I), I, **TOL)
    np.testing.assert_allclose(K.mul(I, I), K.one(), **TOL)
    for k in (1, 2, 3):
        e = K.generator(k)
        np.testing.assert_allclose(K.mul(I, e), K.mul(e, I), **TOL)


def test_cl04_generators():
    """Cl_{0,4}: e_k^2 = -1, e_k e_l = -e_l e_k, e_k^* = -e_k."""
    K = Cl04()
    one = K.one()
    for k in range(1, 5):
        ek = K.generator(k)
        np.testing.assert_allclose(K.mul(ek, ek), -one, **TOL)
        np.testing.assert_allclose(K.star(ek), -ek, **TOL)
        for l in range(k + 1, 5):
            el = K.generator(l)
            np.testing.assert_allclose(K.mul(ek, el), -K.mul(el, ek), **TOL)


def test_quaternion_matrix_algebra_is_entrywise_chi(rng):
    """M_d(H): block[(p, a), (q, c)] = chi(Q[p, q])[a, c]."""
    K = MatrixAlgebra(2, "H")
    x = rng.standard_normal(16)
    (g,) = K.forward(x)
    Q = x.reshape(2, 2, 4)
    blk = g.data[0].reshape(2, 2, 2, 2)
    for p in range(2):
        for q in range(2):
            a, b, c, e = Q[p, q]
            chi = np.array([[a + 1j * b, c + 1j * e], [-c + 1j * e, a - 1j * b]])
            np.testing.assert_allclose(blk[p, :, q, :], chi, **TOL)


def test_dft_matches_circular_convolution(rng):
    K = TubeDFT(6)
    a, b = rng.standard_normal(6), rng.standard_normal(6)
    conv = np.array([sum(a[j] * b[(k - j) % 6] for j in range(6)) for k in range(6)])
    np.testing.assert_allclose(K.mul(a, b), conv, **TOL)


def test_dct_matches_scipy(rng):
    import scipy.fft

    K = TubeDCT(7)
    x = rng.standard_normal((2, 7))
    (g,) = K.forward(x)
    np.testing.assert_allclose(g.data[..., 0, 0], scipy.fft.dct(x, norm="ortho", axis=-1), **TOL)


def test_tensor_product_axes_follow_factor_order(rng):
    # forward/inverse must not depend on where the non-*_M factor sits
    A = TensorProduct(TubeDFT(4), TubeDCT(3))
    B = TensorProduct(TubeDCT(3), TubeDFT(4))
    x = rng.standard_normal((2, 4, 3))
    xa = A.inverse(A.forward(x))
    xb = np.swapaxes(B.inverse(B.forward(np.swapaxes(x, -1, -2))), -1, -2)
    np.testing.assert_allclose(xa, x, **TOL)
    np.testing.assert_allclose(xb, x, **TOL)


def test_non_orthogonal_M_is_rejected():
    from staralg.algebras import StarM

    with pytest.raises(ValueError):
        StarM(np.array([[1.0, 0.0], [1.0, 1.0]]))


def test_two_dft_factors_give_the_product_group_algebra():
    # R[Z_3] (x) R[Z_4] = R[Z_3 x Z_4]  (this used to raise NotImplementedError)
    K = TensorProduct(TubeDFT(3), TubeDFT(4))
    assert isinstance(K, AbelianGroupAlgebra) and K.shape == (3, 4)


def test_group_algebra_times_clifford_is_still_rejected():
    with pytest.raises(NotImplementedError):
        TensorProduct(GroupAlgebra.dihedral(3), Cl10())


def test_data_driven_M_is_orthogonal_and_concentrates_energy(rng):
    from staralg.algebras import StarM

    X = rng.standard_normal((6, 5, 7)) * np.array([5.0, 1.0, 0.5, 0.2, 0.1])[None, :, None]
    K = StarM.from_data(X, axis=1)
    np.testing.assert_allclose(K.M @ K.M.T, np.eye(5), atol=1e-10)
    (g,) = K.forward(np.moveaxis(X, 1, -1))
    energy = np.sum(g.data[..., 0, 0] ** 2, axis=(0, 1))
    assert np.all(np.diff(energy) <= 1e-9)


def test_abelian_group_algebra_mul_is_circular_convolution(rng):
    # brute force: (a b)(g) = sum_h a(h) b(g - h) on Z_3 x Z_4, independent of the FFT
    K = AbelianGroupAlgebra(3, 4)
    a, b = rng.standard_normal((3, 4)), rng.standard_normal((3, 4))
    c = np.zeros((3, 4))
    for g1 in range(3):
        for g2 in range(4):
            c[g1, g2] = sum(a[h1, h2] * b[(g1 - h1) % 3, (g2 - h2) % 4] for h1 in range(3) for h2 in range(4))
    np.testing.assert_allclose(K.mul(a, b), c, **TOL)
    s = np.array([[a[(-g1) % 3, (-g2) % 4] for g2 in range(4)] for g1 in range(3)])
    np.testing.assert_allclose(K.star(a), s, **TOL)


def test_abelian_group_algebra_one_factor_matches_tubedft(rng):
    x = rng.standard_normal((5, 6))
    for ga, gb in zip(AbelianGroupAlgebra(6).forward(x), TubeDFT(6).forward(x)):
        np.testing.assert_allclose(ga.data, gb.data, **TOL)
        np.testing.assert_allclose(ga.weights, gb.weights, **TOL)


def test_abelian_group_algebra_block_count(rng):
    # Z_32 x Z_32: 4 real blocks (m_i in {0, 16}) and (1024 - 4) / 2 = 510 conjugate pairs
    g = AbelianGroupAlgebra(32, 32).forward(np.zeros((32, 32)))
    assert [gr.n_blocks for gr in g] == [4, 510]
