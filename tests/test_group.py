"""Tests of the group algebras (staralg/algebras/group.py).

The generic properties (round trip, multiplicativity, involution, Plancherel weights,
unit, adjoint identities, parameter count, quaternion structure, SVD / TT / Tucker
theorems) are checked for GroupAlgebra and AbelianGroupAlgebra in the test files of
those modules.  Here:

  1. The tensor product of group algebras is the group algebra of the product group:
     TensorProduct(K_G, K_H) returns F[G x H], and its product, involution and unit
     are those of the tensor product algebra:  (a (x) b)(c (x) d) = ac (x) bd.
  2. Both spellings of F[G x H] (TensorProduct and GroupAlgebra.product, or
     AbelianGroupAlgebra) give the same structure map.
  3. The numerical irreps (GroupAlgebra from a multiplication table) and the FFT path
     (TubeDFT, AbelianGroupAlgebra) give the same decompositions: SVD, TT and Tucker
     reconstructions and storage agree, since a decomposition over K does not depend on
     the basis chosen for the blocks.  The same holds for isomorphic groups with
     different element labels (D_3 and S_3; a relabelled Q_8, S_4, D_3 x Z_4).
  4. The block types follow the real tensor products of division algebras,
     C (x)_R C = C (+) C and C (x)_R H = M_2(C), and the Frobenius-Schur indicators.
  5. Multiplication tables that are not groups are rejected.
"""
import itertools

import numpy as np
import pytest

from staralg import (
    AbelianGroupAlgebra,
    GroupAlgebra,
    TensorProduct,
    TubeDFT,
    hosvd_K,
    svd_K,
    tt_svd_K,
)
from staralg.algebras.group import (
    abelian_table,
    check_table,
    cyclic_table,
    dihedral_table,
    symmetric_table,
)

TOL = dict(rtol=1e-9, atol=1e-9)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _random(K, rng, batch=()):
    return K.random(rng, batch)


# ---------------------------------------------------------------------------
# 1. F[G] (x) F[H] = F[G x H]
# ---------------------------------------------------------------------------
PAIRS = [
    (lambda: GroupAlgebra.dihedral(3), lambda: TubeDFT(4)),
    (lambda: GroupAlgebra.quaternion(), lambda: TubeDFT(3)),
    (lambda: GroupAlgebra.dihedral(3), lambda: GroupAlgebra.dihedral(4)),
    (lambda: TubeDFT(4), lambda: TubeDFT(6)),
    (lambda: TubeDFT(4), lambda: AbelianGroupAlgebra(2, 3)),
    (lambda: GroupAlgebra.dihedral(4, field="C"), lambda: TubeDFT(3, "C")),
]


@pytest.mark.parametrize("make", PAIRS, ids=["D_3 x Z_4", "Q_8 x Z_3", "D_3 x D_4", "Z_4 x Z_6",
                                              "Z_4 x (Z_2 x Z_3)", "C[D_4 x Z_3]"])
def test_tensor_product_is_product_group_algebra(make, rng):
    G, H = make[0](), make[1]()
    P = TensorProduct(G, H)
    assert isinstance(P, (GroupAlgebra, AbelianGroupAlgebra))
    assert P.shape == G.shape + H.shape
    assert P.is_commutative == (G.is_commutative and H.is_commutative)
    a1, a2 = _random(G, rng), _random(G, rng)
    b1, b2 = _random(H, rng), _random(H, rng)
    outer = np.multiply.outer
    # (a1 (x) b1)(a2 (x) b2) = a1 a2 (x) b1 b2
    np.testing.assert_allclose(P.mul(outer(a1, b1), outer(a2, b2)), outer(G.mul(a1, a2), H.mul(b1, b2)), **TOL)
    np.testing.assert_allclose(P.star(outer(a1, b1)), outer(G.star(a1), H.star(b1)), **TOL)
    np.testing.assert_allclose(P.one(), outer(G.one(), H.one()), **TOL)


def test_three_factors(rng):
    P = TensorProduct(TubeDFT(2), GroupAlgebra.dihedral(3), TubeDFT(3))
    assert isinstance(P, GroupAlgebra) and P.shape == (2, 6, 3) and P.group_name == "Z_2 x D_3 x Z_3"
    x = P.random(rng, (2,))
    np.testing.assert_allclose(P.inverse(P.forward(x)), x, **TOL)


# ---------------------------------------------------------------------------
# 2. Both spellings give the same algebra
# ---------------------------------------------------------------------------
def test_tensor_product_of_dfts_equals_abelian_group_algebra(rng):
    # the Vlasov algebra of experiment 3: R[Z_32] (x) R[Z_32] = R[Z_32 x Z_32]
    P = TensorProduct(TubeDFT(32), TubeDFT(32))
    A = AbelianGroupAlgebra(32, 32)
    assert isinstance(P, AbelianGroupAlgebra) and P.shape == (32, 32)
    x = rng.standard_normal((3, 32, 32))
    for gp, ga in zip(P.forward(x), A.forward(x)):
        np.testing.assert_allclose(gp.data, ga.data, **TOL)
        np.testing.assert_allclose(gp.weights, ga.weights, **TOL)
    # 4 real blocks ((m, n) with m, n in {0, 16}) and (1024 - 4) / 2 conjugate pairs
    assert [g.n_blocks for g in P.forward(x)] == [4, 510]


def test_tensor_product_and_product_constructor_agree(rng):
    G, H = GroupAlgebra.quaternion(), TubeDFT(3)
    P1, P2 = TensorProduct(G, H), GroupAlgebra.product(G, H)
    assert P1.irrep_types() == P2.irrep_types()
    x = P1.random(rng, (2,))
    for g1, g2 in zip(P1.forward(x), P2.forward(x)):
        np.testing.assert_allclose(g1.data, g2.data, **TOL)


def test_abelian_constructor_takes_fast_path():
    assert isinstance(GroupAlgebra.abelian(4, 6), AbelianGroupAlgebra)
    assert isinstance(GroupAlgebra.product(TubeDFT(4), TubeDFT(6)), AbelianGroupAlgebra)


# ---------------------------------------------------------------------------
# 3. Decompositions do not depend on how the blocks were found
# ---------------------------------------------------------------------------
def _relabel(T, perm):
    """Multiplication table of the same group with element g renamed perm[g]."""
    T2 = np.empty_like(T)
    T2[np.ix_(perm, perm)] = perm[T]
    return T2


def _assert_same_decompositions(K1, K2, to2, rng, commutative):
    """to2 maps coefficient arrays of K1 (trailing axes K1.shape) to K2's."""
    A = K1.random(rng, (5, 4))
    s1, s2 = svd_K(A, K1), svd_K(to2(A), K2)
    for k in (1, 2):
        np.testing.assert_allclose(to2(s1.reconstruct(rank=k)), s2.reconstruct(rank=k), **TOL)
        assert np.isclose(s1.n_params(rank=k), s2.n_params(rank=k))
    budget = 0.3 * A.size
    np.testing.assert_allclose(to2(s1.reconstruct(budget=budget)), s2.reconstruct(budget=budget), **TOL)
    X = K1.random(rng, (3, 4, 3))
    for adaptive in (False, True):
        t1 = tt_svd_K(X, K1, eps=0.4, adaptive=adaptive)
        t2 = tt_svd_K(to2(X), K2, eps=0.4, adaptive=adaptive)
        np.testing.assert_allclose(to2(t1.reconstruct()), t2.reconstruct(), **TOL)
        assert np.isclose(t1.n_params(), t2.n_params())
        if commutative:
            h1 = hosvd_K(X, K1).truncate(eps=0.4, adaptive=adaptive)
            h2 = hosvd_K(to2(X), K2).truncate(eps=0.4, adaptive=adaptive)
            np.testing.assert_allclose(to2(h1.reconstruct()), h2.reconstruct(), **TOL)
            assert np.isclose(h1.n_params(), h2.n_params())


def _block_signature(K):
    z = np.zeros(K.shape, dtype=float if K.field == "R" else complex)
    return sorted((g.division, g.d, float(w)) for g in K.forward(z) for w in g.weights)


@pytest.mark.parametrize("n", [5, 6])
def test_numerical_irreps_match_tubedft(n, rng):
    K1, K2 = TubeDFT(n), GroupAlgebra(cyclic_table(n), group_name=f"Z_{n} from its table")
    assert _block_signature(K1) == _block_signature(K2)
    _assert_same_decompositions(K1, K2, lambda a: a, rng, commutative=True)


def test_numerical_irreps_match_abelian_fft(rng):
    K1 = AbelianGroupAlgebra(4, 3)
    K2 = GroupAlgebra(abelian_table((4, 3)), shape=(4, 3), group_name="Z_4 x Z_3 from its table")
    assert _block_signature(K1) == _block_signature(K2)
    _assert_same_decompositions(K1, K2, lambda a: a, rng, commutative=True)


def test_dihedral_3_and_symmetric_3_agree(rng):
    # D_3 acts on the vertices {0, 1, 2} of a triangle: r^k s^j  ->  i |-> k + (-1)^j i (mod 3)
    perms = list(itertools.permutations(range(3)))
    phi = np.array([perms.index(tuple((k + (-1) ** j * i) % 3 for i in range(3)))
                    for k in range(3) for j in range(2)])
    TD, TS = dihedral_table(3), symmetric_table(3)
    assert np.array_equal(TS[np.ix_(phi, phi)], phi[TD])          # phi is an isomorphism
    K1, K2 = GroupAlgebra.dihedral(3), GroupAlgebra.symmetric(3)

    def to2(a):
        out = np.empty_like(a)
        out[..., phi] = a
        return out

    _assert_same_decompositions(K1, K2, to2, rng, commutative=False)


@pytest.mark.parametrize("make", [GroupAlgebra.quaternion, lambda: GroupAlgebra.symmetric(4),
                                  lambda: GroupAlgebra.product(GroupAlgebra.dihedral(3), TubeDFT(4))],
                         ids=["Q_8", "S_4", "D_3 x Z_4"])
def test_relabelled_group_gives_same_decompositions(make, rng):
    K1 = make()
    n = int(np.prod(K1.shape))
    perm = np.random.default_rng(7).permutation(n)
    K2 = GroupAlgebra(_relabel(K1.cayley_table(), perm), group_name="relabelled")
    assert K1.irrep_types() == K2.irrep_types()

    def to2(a):
        flat = a.reshape(a.shape[: a.ndim - len(K1.shape)] + (n,))
        out = np.empty_like(flat)
        out[..., perm] = flat
        return out

    _assert_same_decompositions(K1, K2, to2, rng, commutative=K1.is_commutative)


# ---------------------------------------------------------------------------
# 4. Block types
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("make, expected", [
    (lambda: GroupAlgebra.dihedral(4), [("R", 1, 4), ("R", 2, 1)]),
    (lambda: GroupAlgebra.dihedral(5), [("R", 1, 2), ("R", 2, 2)]),
    (lambda: GroupAlgebra.symmetric(4), [("R", 1, 2), ("R", 2, 1), ("R", 3, 2)]),
    # Q_8: R[Q_8] = R^4 (+) H  (the 2-dimensional irrep has Frobenius-Schur indicator -1)
    (lambda: GroupAlgebra.quaternion(), [("R", 1, 4), ("H", 1, 1)]),
    (lambda: GroupAlgebra(cyclic_table(5)), [("R", 1, 1), ("C", 1, 2)]),
    # C (x)_R C = C (+) C: the complex blocks of Z_4 times the real 2-dim irrep of D_3 give M_2(C)
    (lambda: TensorProduct(GroupAlgebra.dihedral(3), TubeDFT(4)),
     [("R", 1, 4), ("R", 2, 2), ("C", 1, 2), ("C", 2, 1)]),
    # C (x)_R H = M_2(C): the quaternion block of Q_8 times the complex block of Z_3
    (lambda: TensorProduct(GroupAlgebra.quaternion(), TubeDFT(3)),
     [("R", 1, 4), ("C", 1, 4), ("C", 2, 1), ("H", 1, 1)]),
    # H (x)_R M_2(R) = M_2(H): a 4-dimensional quaternionic irrep (two quaternion entries per
    # side, so the interleaved layout of MatrixAlgebra(2, 'H') is tested)
    (lambda: TensorProduct(GroupAlgebra.quaternion(), GroupAlgebra.dihedral(3)),
     [("R", 1, 8), ("R", 2, 4), ("H", 1, 2), ("H", 2, 1)]),
    # over C every irrep is a complex block
    (lambda: GroupAlgebra.quaternion(field="C"), [("C", 1, 4), ("C", 2, 1)]),
], ids=["D_4", "D_5", "S_4", "Q_8", "Z_5", "D_3 x Z_4", "Q_8 x Z_3", "Q_8 x D_3", "C[Q_8]"])
def test_block_types(make, expected):
    K = make()
    assert K.irrep_types() == expected
    # real dimension: sum over blocks of (number of blocks) x (real dim of M_d(D))
    real = {"R": 1, "C": 2, "H": 4}
    assert sum(B * d * d * real[div] for div, d, B in expected) == K.real_dim


def test_q8_quaternion_block_is_the_quaternions(rng):
    # in the H block of R[Q_8] the group elements i, j, k act as the quaternion units
    K = GroupAlgebra.quaternion()
    e = np.eye(8)
    blocks = [K.forward(e[g])[-1].data[0] for g in (1, 2, 3)]   # i, j, k
    I2 = np.eye(2)
    for q in blocks:
        np.testing.assert_allclose(q @ q, -I2, **TOL)
    np.testing.assert_allclose(blocks[0] @ blocks[1], blocks[2], **TOL)    # i j = k


# ---------------------------------------------------------------------------
# 5. Tables that are not groups
# ---------------------------------------------------------------------------
def test_non_associative_loop_is_rejected():
    # a Latin square with identity 0 in which every element squares to 0: a loop of
    # order 5 that is not a group (a group of order 5 is cyclic)
    L = np.array([[0, 1, 2, 3, 4],
                  [1, 0, 3, 4, 2],
                  [2, 4, 0, 1, 3],
                  [3, 2, 4, 0, 1],
                  [4, 3, 1, 2, 0]])
    with pytest.raises(ValueError, match="associative"):
        check_table(L)


def test_non_latin_square_is_rejected():
    with pytest.raises(ValueError, match="Latin"):
        GroupAlgebra(np.zeros((3, 3), dtype=int))


def test_shape_must_match_order():
    with pytest.raises(ValueError):
        GroupAlgebra(cyclic_table(6), shape=(4, 2))
