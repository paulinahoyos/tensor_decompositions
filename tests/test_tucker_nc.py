"""Tests of Tucker over a noncommutative K (staralg/core/tucker_nc.py, opt-in).

  * by default hosvd_K / sthosvd_K still refuse a noncommutative K;
  * full core size reconstructs exactly (mirror order), for HOSVD and ST-HOSVD, with any gauge;
  * the factors and the core are elements over K (forward(inverse(blocks)) == blocks);
  * error identity for any K: ||X - X_hat||^2 = sum_k tau~_k (energy removed while forming
    the core), for HOSVD and ST-HOSVD; for ST-HOSVD tau~_k are the truncated tails, so
    ||X - X_hat||^2 = sum of tails, and with eps, ||X - X_hat|| <= eps ||X||;
  * storage: p m^2 (prod r + sum n r) per block;
  * on a commutative K the block path agrees with the commutative implementation;
  * the 2 x 1 quaternion example: with U_2 = [j] the HOSVD returns 0, while ST-HOSVD is exact
    (checked with an independent coefficient-level implementation using K.mul).
"""
import numpy as np
import pytest

from staralg import (
    Cl03,
    Cl04,
    DirectSum,
    GroupAlgebra,
    MatrixAlgebra,
    NoncommutativeTuckerError,
    Quaternions,
    TubeDFT,
    hosvd_K,
    sthosvd_K,
)
from staralg.core.tucker_nc import hosvd_nc, sthosvd_nc, ttm_nc

NONCOMMUTATIVE = [
    Quaternions(),
    Cl03(),
    Cl04(),
    MatrixAlgebra(2, "R"),
    MatrixAlgebra(2, "C"),
    GroupAlgebra.dihedral(4),
    GroupAlgebra.quaternion(),
    DirectSum(MatrixAlgebra(2, "C"), MatrixAlgebra(2, "C")),
]
DIMS = (5, 4, 3)
TOL = 1e-10


@pytest.fixture(params=NONCOMMUTATIVE, ids=lambda K: K.name)
def K(request):
    return request.param


def _norm(K, X):
    return float(np.sqrt(K.norm2(X).sum()))


def planted(K, dims, r, rng, noise=0.0):
    """A tensor with an exact Tucker decomposition of core size r (left-to-right convention),
    plus relative Gaussian noise."""
    C = K.random(rng, r)
    g0 = K.forward(C)
    N = len(dims)
    blocks = [np.moveaxis(g.data, N, 0) for g in g0]
    for k, n in enumerate(dims):
        Ug = K.forward(K.random(rng, (n, r[k])))
        blocks = [ttm_nc(T, gu.data.transpose(2, 0, 3, 1, 4).reshape(gu.data.shape[2], n * g.m, r[k] * g.m), k)
                  for T, gu, g in zip(blocks, Ug, g0)]
    X = K.inverse([g.with_data(np.moveaxis(T, 0, N)) for g, T in zip(g0, blocks)])
    if noise:
        X = X + noise * _norm(K, X) / np.sqrt(np.prod(dims)) * K.random(rng, dims)
    return X


def test_default_still_refuses(K):
    X = K.random(np.random.default_rng(0), DIMS)
    with pytest.raises(NoncommutativeTuckerError):
        hosvd_K(X, K)
    with pytest.raises(NoncommutativeTuckerError):
        sthosvd_K(X, K, ranks=1)


@pytest.mark.parametrize("gauge", [None, 5])
def test_full_size_is_exact(K, gauge):
    X = K.random(np.random.default_rng(1), DIMS)
    g = None if gauge is None else np.random.default_rng(gauge)
    T = sthosvd_K(X, K, ranks=DIMS, allow_noncommutative=True, gauge=g)
    assert _norm(K, X - T.reconstruct()) <= TOL * _norm(K, X)
    g = None if gauge is None else np.random.default_rng(gauge)
    H = hosvd_K(X, K, allow_noncommutative=True).truncate(ranks=DIMS, gauge=g)
    assert _norm(K, X - H.reconstruct()) <= TOL * _norm(K, X)


def _pad(arrs, axes):
    R = [max(a.shape[ax] for a in arrs) for ax in axes]
    out = []
    for a in arrs:
        pad = [(0, 0)] * a.ndim
        for ax, r in zip(axes, R):
            pad[ax] = (0, r - a.shape[ax])
        out.append(np.pad(a, pad))
    return out


@pytest.mark.parametrize("adaptive", [False, True])
def test_factors_and_core_are_elements_over_K(K, adaptive):
    X = planted(K, DIMS, (2, 2, 2), np.random.default_rng(2), noise=0.1)
    T = sthosvd_K(X, K, eps=0.3, adaptive=adaptive, allow_noncommutative=True, gauge=np.random.default_rng(3))
    N = len(DIMS)
    scale = max(np.abs(C).max() for C in T.cores)
    for k in range(N):
        D = [F.reshape(F.shape[0], F.shape[1] // g.m, g.m, F.shape[2] // g.m, g.m).transpose(1, 3, 0, 2, 4)
             for F, g in zip([Us[k] for Us in T.factors], T.groups)]
        G = [g.with_data(d) for g, d in zip(T.groups, _pad(D, [1]))]
        for a, b in zip(G, K.forward(K.inverse(G))):
            np.testing.assert_allclose(a.data, b.data, atol=1e-10)
    D = _pad([np.moveaxis(C, 0, N) for C in T.cores], list(range(N)))
    G = [g.with_data(d) for g, d in zip(T.groups, D)]
    for a, b in zip(G, K.forward(K.inverse(G))):
        np.testing.assert_allclose(a.data, b.data, atol=1e-10 * scale)


@pytest.mark.parametrize("data", ["random", "planted"])
@pytest.mark.parametrize("gauge", [None, 7])
def test_sthosvd_error_identity(K, data, gauge):
    rng = np.random.default_rng(4)
    X = K.random(rng, DIMS) if data == "random" else planted(K, DIMS, (2, 2, 2), rng, noise=0.05)
    nx = _norm(K, X)
    for r in [(1, 1, 1), (2, 2, 2), (3, 2, 1), (2, 3, 2)]:
        T = sthosvd_K(X, K, ranks=r, allow_noncommutative=True,
                      gauge=None if gauge is None else np.random.default_rng(gauge))
        err2 = _norm(K, X - T.reconstruct()) ** 2
        np.testing.assert_allclose(T.tails, T.seq_tails, rtol=1e-8, atol=1e-10 * nx ** 2)
        np.testing.assert_allclose(err2, sum(T.tails), rtol=1e-8, atol=1e-12 * nx ** 2)
    for eps in (0.05, 0.2, 0.5):
        for adaptive in (False, True):
            T = sthosvd_K(X, K, eps=eps, adaptive=adaptive, allow_noncommutative=True,
                          gauge=None if gauge is None else np.random.default_rng(gauge))
            assert _norm(K, X - T.reconstruct()) <= eps * nx * (1 + 1e-9)


@pytest.mark.parametrize("gauge", [None, 8])
def test_hosvd_aposteriori_error_identity(K, gauge):
    rng = np.random.default_rng(5)
    X = planted(K, DIMS, (2, 2, 2), rng, noise=0.05)
    nx = _norm(K, X)
    h = hosvd_K(X, K, allow_noncommutative=True)
    for r in [(1, 1, 1), (2, 2, 2), (3, 2, 1)]:
        T = h.truncate(ranks=r, gauge=None if gauge is None else np.random.default_rng(gauge))
        np.testing.assert_allclose(_norm(K, X - T.reconstruct()) ** 2, sum(T.seq_tails),
                                   rtol=1e-8, atol=1e-12 * nx ** 2)


def test_hosvd_error_exceeds_tails_on_ya_yang_tensor():
    # Ya and Yang (arXiv:2309.05211), Example 1: T(a,b,c,d) = 2 / ((a - 3) + b i + c j + d k),
    # a, b, c, d in {1, 2, 3}, core size (2, 2, 2, 2).  The tails of T sum to 0.01926; the
    # HOSVD error exceeds them, the ST-HOSVD error equals its own tails.
    import itertools

    K = Quaternions()
    X = np.zeros((3, 3, 3, 3, 4))
    for a, b, c, d in itertools.product(range(1, 4), repeat=4):
        q = np.array([a - 3, b, c, d], float)
        X[a - 1, b - 1, c - 1, d - 1] = 2 * np.array([q[0], -q[1], -q[2], -q[3]]) / q.dot(q)
    H = hosvd_K(X, K, allow_noncommutative=True).truncate(ranks=2)
    assert np.isclose(sum(H.tails), 0.01926, atol=5e-5)
    assert _norm(K, X - H.reconstruct()) ** 2 > 2 * sum(H.tails)
    S = sthosvd_K(X, K, ranks=2, allow_noncommutative=True)
    np.testing.assert_allclose(_norm(K, X - S.reconstruct()) ** 2, sum(S.tails), rtol=1e-8)


def test_storage_count(K):
    X = K.random(np.random.default_rng(6), DIMS)
    r = (2, 1, 3)
    T = sthosvd_K(X, K, ranks=r, allow_noncommutative=True)
    expected = sum(g.n_blocks * g.real_params_per_entry * g.m ** 2 * (np.prod(r) + np.dot(DIMS, r))
                   for g in T.groups)
    assert np.isclose(T.n_params(), expected)
    # uniform storage is (prod r + sum n r) dim_R K
    assert np.isclose(T.n_params(), (np.prod(r) + np.dot(DIMS, r)) * K.real_dim)


@pytest.mark.parametrize("Kc", [TubeDFT(4), TubeDFT(5, "C")], ids=lambda K: K.name)
def test_block_path_matches_commutative_implementation(Kc):
    X = Kc.random(np.random.default_rng(9), DIMS)
    for r in [(1, 1, 1), (2, 2, 2), (3, 2, 1)]:
        a = sthosvd_K(X, Kc, ranks=r, order=[2, 1, 0])
        b = sthosvd_nc(X, Kc, ranks=r)                      # default order 2, 1, 0
        np.testing.assert_allclose(a.reconstruct(), b.reconstruct(), atol=1e-10)
        assert np.isclose(a.n_params(), b.n_params())
        c = hosvd_K(X, Kc).truncate(ranks=r)
        d = hosvd_nc(X, Kc).truncate(ranks=r)
        np.testing.assert_allclose(c.reconstruct(), d.reconstruct(), atol=1e-10)
        # commutative: ST-HOSVD is Pythagorean
        np.testing.assert_allclose(np.linalg.norm((X - b.reconstruct()).ravel()) ** 2, sum(b.seq_tails),
                                   rtol=1e-8)


# ---------------------------------------------------------------------------
# The 2 x 1 quaternion example, with an independent coefficient-level implementation
# ---------------------------------------------------------------------------
def _ttm_coef(K, X, U, k):
    """Mode-k product in coefficient form, entry by entry with K.mul (U from the left)."""
    Xk = np.moveaxis(X, k, 0)
    out = np.zeros((U.shape[0],) + Xk.shape[1:], dtype=np.result_type(X, U))
    for j in range(U.shape[0]):
        for i in range(U.shape[1]):
            out[j] = out[j] + K.mul(np.broadcast_to(U[j, i], Xk[i].shape), Xk[i])
    return np.moveaxis(out, 0, k)


def _h(a=0, b=0, c=0, d=0):
    return np.array([a, b, c, d], dtype=float)


def _ct(K, U):
    """Conjugate transpose of a coefficient-form matrix over K."""
    return np.swapaxes(K.star(U), 0, 1)


def test_two_by_one_quaternion_example():
    K = Quaternions()
    A = np.stack([_h(1), _h(b=1)])[:, None, :]               # [1; i] as a 2 x 1 matrix over H
    U1 = np.stack([_h(1), _h(b=1)])[:, None, :] / np.sqrt(2)  # [1; i] / sqrt 2
    U2 = _h(c=1)[None, None, :]                              # [j]
    # both are leading left singular vectors and nothing is truncated
    for U, k in ((U1, 0), (U2, 1)):
        P = _ttm_coef(K, _ttm_coef(K, A, _ct(K, U), k), U, k)  # U U^H applied in mode k
        np.testing.assert_allclose(P, A, atol=1e-12)
    # HOSVD with these factors, core in mirror order: the reconstruction is 0
    core = _ttm_coef(K, _ttm_coef(K, A, _ct(K, U2), 1), _ct(K, U1), 0)
    T = _ttm_coef(K, _ttm_coef(K, core, U1, 0), U2, 1)
    np.testing.assert_allclose(T, 0 * A, atol=1e-12)
    # with U2 = [1] instead, the HOSVD is exact
    one = _h(1)[None, None, :]
    core = _ttm_coef(K, _ttm_coef(K, A, _ct(K, one), 1), _ct(K, U1), 0)
    T = _ttm_coef(K, _ttm_coef(K, core, U1, 0), one, 1)
    np.testing.assert_allclose(T, A, atol=1e-12)
    # ST-HOSVD keeps U2 = [j] but takes U1 from the partially compressed core: exact
    C2 = _ttm_coef(K, A, _ct(K, U2), 1)                       # [-j; k]
    np.testing.assert_allclose(C2[:, 0], np.stack([_h(c=-1), _h(d=1)]), atol=1e-12)
    V1 = C2 / np.sqrt(2)
    core = _ttm_coef(K, C2, _ct(K, V1), 0)
    T = _ttm_coef(K, _ttm_coef(K, core, V1, 0), U2, 1)
    np.testing.assert_allclose(T, A, atol=1e-12)
    # and the library's ST-HOSVD at size (1, 1) is exact for every gauge
    for s in range(5):
        Tl = sthosvd_K(A, K, ranks=(1, 1), allow_noncommutative=True, gauge=np.random.default_rng(s))
        np.testing.assert_allclose(Tl.reconstruct(), A, atol=1e-12)


def test_planted_structure_is_invisible_to_early_unfoldings():
    # over H, an exact Tucker tensor of core size (2, 2, 2) has unfoldings of full K-rank in
    # modes 1 and 2; only the last mode (the outermost factor) has K-rank 2
    from staralg.core.tucker_nc import unfold_nc

    K = Quaternions()
    X = planted(K, (6, 5, 4), (2, 2, 2), np.random.default_rng(10))
    (g,) = K.forward(X)
    T = np.moveaxis(g.data, 3, 0)
    ranks = []
    for k in range(3):
        s = np.linalg.svd(unfold_nc(T, k), compute_uv=False)[0]
        ranks.append(int(np.ceil((s > 1e-9 * s[0]).sum() / g.m)))
    assert ranks == [6, 5, 2]
