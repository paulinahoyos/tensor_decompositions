"""Real Clifford algebras Cl_{p,q} with the 'Hermitian' involution.

Basis.  Generators e_1, ..., e_n (n = p + q) with e_k^2 = +1 for k <= p and
e_k^2 = -1 for k > p.  The coefficient axis lists the 2^n basis blades in grade
order, e.g. for n = 3:

    (1, e1, e2, e3, e12, e13, e23, e123).

Involution.  e_k^* = e_k^{-1} (Marchuk and Shirokov, "Unitary spaces on Clifford
algebras", 2008), extended anti-multiplicatively.  On a blade of grade r it is
the sign  prod_k (e_k^2) * (-1)^{r(r-1)/2}.  For Cl_{p,0} this is the reversion,
for Cl_{0,q} the Clifford conjugation.  With this involution the Euclidean inner
product on blade coefficients equals <a, b> = scalar part of a^* b, and it
satisfies the adjoint identities.

Structure maps implemented
--------------------------
Cl10  ~ R (+) R           e |-> (+1, -1)                         (split-complex numbers)
Cl03  ~ H (+) H           e_k |-> (-i_k, +i_k), i_k = (i, j, k)  (e123 |-> (+1, -1))
Cl30  ~ M_2(C)            e_k |-> Pauli matrix sigma_k            (e123 |-> i I)
Cl04  ~ M_2(H)            see Cl04()

For Cl03 the central projectors (1 +- e123)/2 split x = u + e123 b (u a vector,
b in the bivector slots) into the quaternions -(u + b) and (u - b): the Elsaesser
variables of MHD when u is the velocity and b the magnetic field in Alfven units.
"""
from __future__ import annotations

import numpy as np

from .base import Group, OneAxisAlgebra

# complex 2x2 realization of the quaternion units i, j, k
_QI = np.array([[1j, 0], [0, -1j]])
_QJ = np.array([[0, 1], [-1, 0]], dtype=complex)
_QK = np.array([[0, 1j], [1j, 0]])

# Pauli matrices
_S1 = np.array([[0, 1], [1, 0]], dtype=complex)
_S2 = np.array([[0, -1j], [1j, 0]])
_S3 = np.array([[1, 0], [0, -1]], dtype=complex)


def _popcount(m: int) -> int:
    return bin(m).count("1")


def _bits(m: int) -> list[int]:
    return [k for k in range(m.bit_length()) if (m >> k) & 1]


def _reorder_sign(a: int, b: int) -> int:
    """Sign from moving the generators of blade b past those of blade a."""
    a >>= 1
    s = 0
    while a:
        s += _popcount(a & b)
        a >>= 1
    return -1 if s & 1 else 1


class Clifford(OneAxisAlgebra):
    """Cl_{p,q} with an explicit structure isomorphism given by generator images.

    Parameters
    ----------
    p, q : int
        Signature: p generators square to +1, q to -1.
    blocks : list of (division, d, images)
        One entry per block of the structure isomorphism.  ``images`` is a list of
        n square matrices, the images of e_1, ..., e_n in that block (stored
        realization: quaternion blocks as complex 2x2 matrices).
        All blocks must have the same division, size and dtype.
    name : str
    """

    is_diagonal_real = False

    def __init__(self, p: int, q: int, blocks, name: str | None = None):
        self.p, self.q = int(p), int(q)
        n = self.p + self.q
        self.n_gen = n
        self.sq = [1] * self.p + [-1] * self.q
        masks = sorted(range(2 ** n), key=lambda m: (_popcount(m), _bits(m)))
        self.masks = masks
        self.index_of_mask = {m: i for i, m in enumerate(masks)}
        self.shape = (2 ** n,)
        self.field = "R"
        self.name = name or f"Cl_{{{p},{q}}}"

        divisions = {b[0] for b in blocks}
        ds = {b[1] for b in blocks}
        if len(divisions) != 1 or len(ds) != 1:
            raise ValueError("all blocks of a Clifford structure map must have the same type")
        self._division = divisions.pop()
        self._d = ds.pop()

        # images of every basis blade in every block: Psi[b, i] = psi_b(blade_i)
        imgs = [[np.asarray(g) for g in b[2]] for b in blocks]
        if any(len(im) != n for im in imgs):
            raise ValueError(f"each block needs images of all {n} generators")
        is_complex = any(np.iscomplexobj(g) for im in imgs for g in im)
        m = imgs[0][0].shape[0]
        dtype = complex if is_complex else float
        Psi = np.zeros((len(blocks), 2 ** n, m, m), dtype=dtype)
        for b, im in enumerate(imgs):
            for i, mask in enumerate(masks):
                P = np.eye(m, dtype=dtype)
                for k in _bits(mask):  # ascending generator order
                    P = P @ im[k]
                Psi[b, i] = P
        self.Psi = Psi
        self._weights = np.full(len(blocks), 1.0 / (len(blocks) * m))

        # real-linear inverse of the structure map
        T = Psi.transpose(1, 0, 2, 3).reshape(2 ** n, -1)  # (2^n, B*m*m)
        if is_complex:
            T = np.concatenate([T.real, T.imag], axis=1)
        self._T = T
        self._Tpinv = np.linalg.pinv(T)
        self._is_complex = is_complex

        # commutativity from the structure constants
        G = self.gamma()
        self.is_commutative = bool(np.allclose(G, G.transpose(1, 0, 2)))

    # ---- basis bookkeeping -------------------------------------------------
    def blade_labels(self) -> list[str]:
        return ["1" if m == 0 else "e" + "".join(str(k + 1) for k in _bits(m)) for m in self.masks]

    def basis(self, label: str) -> np.ndarray:
        """Coefficient vector of a basis blade, e.g. basis('e13')."""
        labels = self.blade_labels()
        v = np.zeros(self.n)
        v[labels.index(label)] = 1.0
        return v

    def generator(self, k: int) -> np.ndarray:
        """Coefficient vector of e_k (k = 1, ..., p+q)."""
        v = np.zeros(self.n)
        v[self.index_of_mask[1 << (k - 1)]] = 1.0
        return v

    def pseudoscalar(self) -> np.ndarray:
        v = np.zeros(self.n)
        v[self.index_of_mask[2 ** self.n_gen - 1]] = 1.0
        return v

    # ---- structure map -------------------------------------------------------
    def forward(self, x):
        x = np.asarray(x)
        self._check_shape(x)
        data = np.einsum("...i,bimn->...bmn", x, self.Psi)
        return [Group(data, self._division, self._d, self._weights)]

    def inverse(self, groups):
        data = groups[0].data
        batch = data.shape[:-3]
        # order (i, b, m, n) flattening must match T: T was built from Psi with axes (i, b, m, n)
        v = data.reshape(batch + (-1,))
        if self._is_complex:
            v = np.concatenate([v.real, v.imag], axis=-1)
        else:
            v = np.real(v)
        return v @ self._Tpinv

    # ---- algebra operations ----------------------------------------------------
    def _gamma(self):
        N = 2 ** self.n_gen
        G = np.zeros((N, N, N))
        for i, a in enumerate(self.masks):
            for j, b in enumerate(self.masks):
                s = _reorder_sign(a, b)
                for k in _bits(a & b):
                    s *= self.sq[k]
                G[i, j, self.index_of_mask[a ^ b]] = s
        return G

    def _star_matrix(self):
        signs = []
        for m in self.masks:
            r = _popcount(m)
            s = (-1) ** (r * (r - 1) // 2)
            for k in _bits(m):
                s *= self.sq[k]
            signs.append(s)
        return np.diag(np.array(signs, dtype=float))

    def one(self):
        v = np.zeros(self.n)
        v[0] = 1.0
        return v

    # ---- encoding of physical fields (n = 3) -----------------------------------
    def _dual_vectors(self) -> np.ndarray:
        """Coefficient vectors of I e_k (I the pseudoscalar), k = 1, ..., n_gen."""
        I = self.pseudoscalar()
        return np.stack([self.mul(I, self.generator(k)) for k in range(1, self.n_gen + 1)])

    def encode(self, vector=None, dual=None, scalar=None, pseudo=None, comp_axis: int = -1):
        """Build x = scalar + sum_k v_k e_k + I (pseudo + sum_k w_k e_k).

        ``vector`` and ``dual`` are arrays with a component axis of length p+q at
        position ``comp_axis``; ``scalar`` and ``pseudo`` are arrays without it.
        The result has the coefficient axis (length 2^(p+q)) last.
        """
        parts = [a for a in (vector, dual) if a is not None]
        if parts:
            batch = np.moveaxis(np.asarray(parts[0]), comp_axis, -1).shape[:-1]
        else:
            batch = np.shape(scalar if scalar is not None else pseudo)
        dtype = np.result_type(*(np.asarray(a) for a in (vector, dual, scalar, pseudo) if a is not None))
        x = np.zeros(batch + (self.n,), dtype=dtype)
        if vector is not None:
            v = np.moveaxis(np.asarray(vector), comp_axis, -1)
            E = np.stack([self.generator(k) for k in range(1, self.n_gen + 1)])
            x = x + v @ E
        if dual is not None:
            w = np.moveaxis(np.asarray(dual), comp_axis, -1)
            x = x + w @ self._dual_vectors()
        if scalar is not None:
            x = x + np.asarray(scalar)[..., None] * self.one()
        if pseudo is not None:
            x = x + np.asarray(pseudo)[..., None] * self.pseudoscalar()
        return x

    def decode(self, x, comp_axis: int = -1) -> dict:
        """Inverse of ``encode``: returns scalar, vector, dual, pseudo."""
        E = np.stack([self.generator(k) for k in range(1, self.n_gen + 1)])
        W = self._dual_vectors()  # rows are +-1 multiples of basis blades
        out = {
            "scalar": x[..., 0],
            "vector": np.moveaxis(x @ E.T, -1, comp_axis),
            "dual": np.moveaxis(x @ W.T, -1, comp_axis),
            "pseudo": x @ self.pseudoscalar(),
        }
        return out


def Cl10() -> Clifford:
    """Split-complex numbers R[e]/(e^2 - 1) ~ R (+) R; blocks carry x0 + x1 and x0 - x1."""
    return Clifford(1, 0, [("R", 1, [np.array([[1.0]])]),
                           ("R", 1, [np.array([[-1.0]])])], name="Cl_{1,0}")


def Cl03() -> Clifford:
    """Cl_{0,3} ~ H (+) H with blocks P_+ (e123 = +1) and P_- (e123 = -1)."""
    plus = [-_QI, -_QJ, -_QK]
    minus = [_QI, _QJ, _QK]
    return Clifford(0, 3, [("H", 1, plus), ("H", 1, minus)], name="Cl_{0,3}")


def Quaternions() -> Clifford:
    """H = Cl_{0,2}: basis (1, e1, e2, e12) = (1, i, j, k).

    A pure quaternion field (v_x, v_y, v_z) has coefficients (0, v_x, v_y, v_z).
    """
    return Clifford(0, 2, [("H", 1, [_QI, _QJ])], name="H")


def Cl04() -> Clifford:
    """Cl_{0,4} ~ M_2(H), one block of quaternion 2x2 matrices (stored as 4x4 complex).

    Generator images (anticommuting, anti-Hermitian, unitary, squaring to -1):
        e1 -> [[0, 1], [-1, 0]],  e2 -> diag(i, -i),  e3 -> diag(j, -j),  e4 -> diag(k, -k),
    realized entrywise with chi (the interleaved convention of MatrixAlgebra).
    """
    D = np.diag([1.0, -1.0])
    J2 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    images = [np.kron(J2, np.eye(2)).astype(complex), np.kron(D, _QI), np.kron(D, _QJ), np.kron(D, _QK)]
    return Clifford(0, 4, [("H", 2, images)], name="Cl_{0,4}")


def Cl30() -> Clifford:
    """Cl_{3,0} ~ M_2(C) via the Pauli matrices (the signature used in Clifford neural networks)."""
    return Clifford(3, 0, [("C", 2, [_S1, _S2, _S3])], name="Cl_{3,0}")
