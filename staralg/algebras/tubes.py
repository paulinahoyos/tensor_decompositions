"""Tube algebras: the t-product (DFT) and the *_M products with an orthogonal M.

TubeDFT(n, field)
    K = F[Z_n], tubes of length n with circular convolution (the t-product).
    Structure map: the unnormalized DFT, a |-> (sum_j a_j w^{jk})_k, w = exp(-2 pi i / n).
    * field 'R': blocks k = 0 (and n/2 for even n) are real; the others come in
      conjugate pairs k, n-k, and only k = 1, ..., ceil(n/2) - 1 are stored.
    * field 'C': all n blocks are complex.
    Involution: a*_j = conj(a_{-j}).

StarM(M)
    K = R^n with a *_M b = M^T (Ma (.) Mb) for an orthogonal M (Kernfeld, Kilmer,
    Aeron 2015; Kilmer, Horesh, Avron, Newman 2021).  Structure map: a |-> Ma, n
    real 1x1 blocks.  Involution: identity.  Special cases: TubeDCT (orthonormal
    DCT-II), WalshHadamard (normalized Sylvester-Hadamard matrix, the group algebra
    of (Z_2)^k), RandomOrthogonal.

TubeDFT(n) is the group algebra of the cyclic group Z_n; group algebras of other finite
groups (products of cyclic groups, dihedral, quaternion, symmetric groups, ...) are in
group.py.
"""
from __future__ import annotations

import numpy as np
import scipy.fft
import scipy.linalg

from .base import Group, OneAxisAlgebra


class TubeDFT(OneAxisAlgebra):
    """F[Z_n] with the DFT as structure map (the t-product of Kilmer and Martin)."""

    is_diagonal_real = False
    is_group_algebra = True

    def cayley_table(self):
        """Multiplication table of Z_n on the coordinate indices: g h = (g + h) mod n."""
        i = np.arange(self.n)
        return (i[:, None] + i[None, :]) % self.n

    def __init__(self, n: int, field: str = "R"):
        if field not in ("R", "C"):
            raise ValueError("field must be 'R' or 'C'")
        self.shape = (int(n),)
        self.field = field
        self.is_commutative = True
        self.name = f"{field}[Z_{n}]"
        self.group_name = f"Z_{n}"
        n = self.n
        self._real_idx = [0] + ([n // 2] if n % 2 == 0 and n > 1 else [])
        self._pair_idx = list(range(1, (n + 1) // 2))

    # ---- structure map -----------------------------------------------------
    def forward(self, x):
        x = np.asarray(x)
        self._check_shape(x)
        n = self.n
        xh = np.fft.fft(x, axis=-1)
        if self.field == "C":
            return [Group(xh[..., :, None, None], "C", 1, np.full(n, 1.0 / n))]
        groups = [Group(np.ascontiguousarray(xh[..., self._real_idx].real)[..., None, None],
                        "R", 1, np.full(len(self._real_idx), 1.0 / n))]
        if self._pair_idx:
            groups.append(Group(xh[..., self._pair_idx][..., None, None],
                                "C", 1, np.full(len(self._pair_idx), 2.0 / n)))
        return groups

    def inverse(self, groups):
        n = self.n
        if self.field == "C":
            return np.fft.ifft(groups[0].data[..., 0, 0], axis=-1)
        real_part = np.real(groups[0].data[..., 0, 0])
        xh = np.zeros(real_part.shape[:-1] + (n,), dtype=complex)
        xh[..., self._real_idx] = real_part
        if self._pair_idx:
            c = groups[1].data[..., 0, 0]
            idx = np.array(self._pair_idx)
            xh[..., idx] = c
            xh[..., n - idx] = np.conj(c)
        return np.fft.ifft(xh, axis=-1).real

    # ---- algebra operations (tests) ------------------------------------------
    def _gamma(self):
        n = self.n
        G = np.zeros((n, n, n))
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        G[i, j, (i + j) % n] = 1.0
        return G

    def _star_matrix(self):
        n = self.n
        S = np.zeros((n, n))
        S[np.arange(n), (-np.arange(n)) % n] = 1.0
        return S

    def one(self):
        e = np.zeros(self.n, dtype=float if self.field == "R" else complex)
        e[0] = 1.0
        return e


class StarM(OneAxisAlgebra):
    """R^n with the *_M product for an orthogonal matrix M."""

    is_diagonal_real = True

    def __init__(self, M, name: str = "StarM", check: bool = True):
        M = np.asarray(M, dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("M must be a square matrix")
        if check and not np.allclose(M @ M.T, np.eye(M.shape[0]), atol=1e-10):
            raise ValueError(
                "M must be orthogonal: the optimality results (Theorem 4.2) need the "
                "Euclidean inner product to satisfy the adjoint identities"
            )
        self.M = M
        self.shape = (M.shape[0],)
        self.field = "R"
        self.is_commutative = True
        self.name = f"{name}_{M.shape[0]}"

    @classmethod
    def from_data(cls, X, axis: int, name: str = "DataM") -> "StarM":
        """Data-driven M (Kilmer, Horesh, Avron, Newman 2021): the rows of M are the left
        singular vectors of the mode-``axis`` unfolding of the real array X, in decreasing
        order of singular value, so the first blocks carry most of the energy."""
        X = np.asarray(X)
        if np.iscomplexobj(X):
            raise ValueError("StarM.from_data needs real data (StarM is a real algebra)")
        A = np.moveaxis(X, axis, 0).reshape(X.shape[axis], -1).astype(np.float64)
        w, V = np.linalg.eigh(A @ A.T)
        M = V[:, ::-1].T
        return cls(M, name=name, check=False)

    # transform along the last axis; subclasses may override with a fast transform
    def _transform(self, x):
        return x @ self.M.T

    def _itransform(self, y):
        return y @ self.M

    def forward(self, x):
        x = np.asarray(x)
        self._check_shape(x)
        y = self._transform(x)
        return [Group(y[..., :, None, None], "R", 1, np.ones(self.n))]

    def inverse(self, groups):
        return self._itransform(groups[0].data[..., 0, 0])

    def _gamma(self):
        M = self.M
        return np.einsum("ri,rj,rk->ijk", M, M, M)

    def _star_matrix(self):
        return np.eye(self.n)

    def one(self):
        return self.M.T @ np.ones(self.n)


class TubeDCT(StarM):
    """*_M product with the orthonormal DCT-II (the c-product)."""

    def __init__(self, n: int):
        M = scipy.fft.dct(np.eye(n), type=2, norm="ortho", axis=0)
        super().__init__(M, name="DCT", check=False)

    def _transform(self, x):
        return scipy.fft.dct(x, type=2, norm="ortho", axis=-1)

    def _itransform(self, y):
        return scipy.fft.idct(y, type=2, norm="ortho", axis=-1)


class WalshHadamard(StarM):
    """*_M product with the normalized Sylvester-Hadamard matrix (n a power of 2).

    This is the real group algebra of (Z_2)^k, n = 2^k.
    """

    def __init__(self, n: int):
        if n < 1 or n & (n - 1):
            raise ValueError("n must be a power of 2")
        super().__init__(scipy.linalg.hadamard(n) / np.sqrt(n), name="WH", check=False)


class RandomOrthogonal(StarM):
    """*_M product with a random orthogonal M (a structure-agnostic baseline)."""

    def __init__(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        Q, R = np.linalg.qr(rng.standard_normal((n, n)))
        Q = Q * np.sign(np.diag(R))
        super().__init__(Q, name=f"RandOrth(seed={seed})", check=False)

