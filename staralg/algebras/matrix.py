"""Full matrix algebras M_d(D), D in {R, C, H}, with the conjugate transpose as involution.

Coefficients.  One axis, ordered row-major over the entries (p, q):
  * D = R: d^2 real coordinates A[p, q]                         (field R)
  * D = C: d^2 complex coordinates A[p, q]                      (field C)
  * D = H: 4 d^2 real coordinates (a, b, c, e) of each entry
           A[p, q] = a + b i + c j + e k, ordered (p, q, unit)    (field R)
The Euclidean inner product of the coefficients is Re tr(A^H B) (or tr(A^H B) over C),
so the adjoint identities hold.

Structure map: the identity, one block.  A quaternion matrix is stored in its complex
realization, entry by entry (the interleaved convention of the whole package):

    [Q]_{(p, a), (q, c)} = chi(Q[p, q])[a, c],
    chi(a + b i + c j + e k) = [[a + b i,  c + e i],
                                [-c + e i, a - b i]],

a 2d x 2d complex matrix.  Since ||chi(q)||_F^2 = 2 |q|^2, the block weight is 1/2 for
D = H and 1 otherwise.
"""
from __future__ import annotations

import numpy as np

from .base import Group, OneAxisAlgebra

# chi of the quaternion units 1, i, j, k
_CHI = np.array([
    [[1, 0], [0, 1]],
    [[1j, 0], [0, -1j]],
    [[0, 1], [-1, 0]],
    [[0, 1j], [1j, 0]],
], dtype=complex)

# quaternion unit products: _QMUL[u, v] = (sign, w) with e_u e_v = sign e_w
_QMUL = {
    (0, 0): (1, 0), (0, 1): (1, 1), (0, 2): (1, 2), (0, 3): (1, 3),
    (1, 0): (1, 1), (1, 1): (-1, 0), (1, 2): (1, 3), (1, 3): (-1, 2),
    (2, 0): (1, 2), (2, 1): (-1, 3), (2, 2): (-1, 0), (2, 3): (1, 1),
    (3, 0): (1, 3), (3, 1): (1, 2), (3, 2): (-1, 1), (3, 3): (-1, 0),
}


class MatrixAlgebra(OneAxisAlgebra):
    """M_d(D) for D in {'R', 'C', 'H'}."""

    is_diagonal_real = False

    def __init__(self, d: int, division: str = "R"):
        if division not in ("R", "C", "H"):
            raise ValueError("division must be 'R', 'C' or 'H'")
        self.d = int(d)
        self.division = division
        self.field = "C" if division == "C" else "R"
        self.is_commutative = self.d == 1 and division != "H"
        self.shape = ((4 if division == "H" else 1) * self.d ** 2,)
        self.name = f"M_{self.d}({division})"
        # coefficient vector -> stored block (linear map), as an array (n_coef, m, m)
        d = self.d
        if division == "H":
            P = np.zeros((d, d, 4, 2 * d, 2 * d), dtype=complex)
            for p in range(d):
                for q in range(d):
                    E = np.zeros((d, d))
                    E[p, q] = 1.0
                    for u in range(4):
                        P[p, q, u] = np.kron(E, _CHI[u])
            self._basis_blocks = P.reshape(4 * d * d, 2 * d, 2 * d)
        else:
            P = np.zeros((d, d, d, d))
            for p in range(d):
                for q in range(d):
                    P[p, q, p, q] = 1.0
            self._basis_blocks = P.reshape(d * d, d, d)

    # ---- structure map -----------------------------------------------------
    def forward(self, x):
        x = np.asarray(x)
        self._check_shape(x)
        d = self.d
        if self.division == "H":
            data = np.einsum("...i,imn->...mn", x, self._basis_blocks)
            w = 0.5
        else:
            data = x.reshape(x.shape[:-1] + (d, d))
            w = 1.0
        return [Group(data[..., None, :, :], self.division, d, np.array([w]))]

    def inverse(self, groups):
        data = groups[0].data[..., 0, :, :]
        d = self.d
        if self.division != "H":
            out = data.reshape(data.shape[:-2] + (d * d,))
            return np.real(out) if self.division == "R" else out
        # read (a, b, c, e) off the first row of each 2x2 sub-block
        blk = data.reshape(data.shape[:-2] + (d, 2, d, 2))
        alpha, beta = blk[..., :, 0, :, 0], blk[..., :, 0, :, 1]
        coef = np.stack([alpha.real, alpha.imag, beta.real, beta.imag], axis=-1)  # (..., d, d, 4)
        return coef.reshape(coef.shape[:-3] + (4 * d * d,))

    # ---- algebra operations (tests) ------------------------------------------
    def _gamma(self):
        d, n = self.d, self.n
        G = np.zeros((n, n, n))
        if self.division != "H":
            for p in range(d):
                for q in range(d):
                    for s in range(d):
                        G[p * d + q, q * d + s, p * d + s] = 1.0
            return G
        idx = lambda p, q, u: (p * d + q) * 4 + u  # noqa: E731
        for p in range(d):
            for q in range(d):
                for s in range(d):
                    for u in range(4):
                        for v in range(4):
                            sign, w = _QMUL[(u, v)]
                            G[idx(p, q, u), idx(q, s, v), idx(p, s, w)] += sign
        return G

    def _star_matrix(self):
        d, n = self.d, self.n
        S = np.zeros((n, n))
        for p in range(d):
            for q in range(d):
                if self.division != "H":
                    S[q * d + p, p * d + q] = 1.0
                else:
                    for u, sign in enumerate((1, -1, -1, -1)):  # quaternion conjugate
                        S[(q * d + p) * 4 + u, (p * d + q) * 4 + u] = sign
        return S

    def one(self):
        d = self.d
        e = np.zeros(self.n, dtype=complex if self.field == "C" else float)
        for p in range(d):
            e[(p * d + p) * (4 if self.division == "H" else 1)] = 1.0
        return e
