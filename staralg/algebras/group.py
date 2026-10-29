"""Group algebras F[G] of finite groups G.

Elements are functions a: G -> F, stored as coefficient arrays indexed by the group
elements; the product is convolution over G and the involution is

    a*(g) = conj(a(g^{-1})),

so the Euclidean inner product on coefficients satisfies the adjoint identities.

Structure map.  The Fourier transform over G,

    a |-> ( sum_g a(g) rho(g) )_rho,

over the irreducible unitary representations rho of G (one block per irrep, of size
d_rho).  Plancherel:  ||a||^2 = sum_rho (d_rho / |G|) ||rho(a)||_F^2.

Real form (field 'R').  The block of an irrep is real, complex or quaternionic according
to its Frobenius-Schur indicator  nu = (1/|G|) sum_g chi(g^2):
    nu = +1  real type:          rho is realized by real orthogonal matrices, block M_d(R),
                                 weight d/|G|
    nu =  0  complex type:       rho is not equivalent to conj(rho); the pair is stored once
                                 as M_d(C), weight 2d/|G|
    nu = -1  quaternionic type:  rho is realized in the interleaved complex form of
                                 M_{d/2}(H) (each quaternion z + w j as [[z, w], [-conj(w), conj(z)]]),
                                 weight d/|G|
With field 'C' every irrep gives one block M_d(C) with weight d/|G|.

Classes
-------
AbelianGroupAlgebra(n_1, ..., n_k, field)
    F[Z_{n_1} x ... x Z_{n_k}] (every finite abelian group is of this form), with the
    k-dimensional FFT as structure map: one 1 x 1 block per wavevector.  Coordinates on
    k axes (e.g. x and y of a periodic box).  The fast path for abelian groups; for k = 1
    it equals TubeDFT(n).
GroupAlgebra(table, shape, field, group_name)
    Any finite group, given by its multiplication table on the indices 0, ..., |G| - 1
    (table[g, h] = index of g h).  The irreps are computed numerically from the table
    (see _complex_irreps): no character table needs to be supplied.  Meant for groups of
    order up to about 2000; use AbelianGroupAlgebra for large abelian groups.
    Constructors:
        GroupAlgebra.abelian(n_1, ..., n_k)   -> AbelianGroupAlgebra (fast path)
        GroupAlgebra.dihedral(n)              D_n, order 2n
        GroupAlgebra.quaternion()             Q_8 (has a quaternionic irrep: R[Q_8] = R^4 (+) H)
        GroupAlgebra.symmetric(n)             S_n, order n!
        GroupAlgebra.product(K_1, ..., K_m)   G_1 x ... x G_m from group algebras K_i

Tensor products.  F[G x H] = F[G] (x) F[H].  TensorProduct of group algebras returns
GroupAlgebra.product of them (an AbelianGroupAlgebra if every factor is abelian), so
both spellings give the same algebra.  Coordinates: the axes of the factors in order,
so a coefficient array a[g, h] is indexed like the tensor product a_G (x) a_H.

Group-algebra protocol (TubeDFT, AbelianGroupAlgebra, GroupAlgebra): the attribute
is_group_algebra = True, group_name, and cayley_table() (the flat multiplication table,
elements in C order of the coordinate shape).
"""
from __future__ import annotations

import itertools

import numpy as np

from .base import Algebra, Group

MAX_ORDER = 2000          # GroupAlgebra computes |G| x |G| eigendecompositions


# ---------------------------------------------------------------------------
# Multiplication tables (flat indices; table[g, h] = index of g h)
# ---------------------------------------------------------------------------
def cyclic_table(n: int) -> np.ndarray:
    """Z_n:  g h = (g + h) mod n."""
    i = np.arange(n)
    return (i[:, None] + i[None, :]) % n


def abelian_table(ns) -> np.ndarray:
    """Z_{n_1} x ... x Z_{n_k}, elements in C order of the grid (n_1, ..., n_k)."""
    ns = tuple(int(n) for n in ns)
    N = int(np.prod(ns))
    multi = np.unravel_index(np.arange(N), ns)
    sums = tuple((m[:, None] + m[None, :]) % n for m, n in zip(multi, ns))
    return np.ravel_multi_index(sums, ns)


def dihedral_table(n: int) -> np.ndarray:
    """D_n = <r, s | r^n = s^2 = 1, s r s = r^{-1}>, order 2n.  Element r^k s^j has index
    2k + j;  (r^a s^b)(r^c s^d) = r^{a + (-1)^b c} s^{b + d}."""
    k, j = np.divmod(np.arange(2 * n), 2)
    a, b, c, d = k[:, None], j[:, None], k[None, :], j[None, :]
    kk = (a + np.where(b == 0, c, -c)) % n
    return 2 * kk + (b + d) % 2


def quaternion_table() -> np.ndarray:
    """Q_8 = {+-1, +-i, +-j, +-k}.  Element s * u (s = +-1, u in (1, i, j, k)) has index
    4 * (0 if s = +1 else 1) + (0, 1, 2, 3)[u]."""
    # u v = sign * w for the units 1, i, j, k (indices 0..3)
    unit = {(0, 0): (1, 0), (0, 1): (1, 1), (0, 2): (1, 2), (0, 3): (1, 3),
            (1, 0): (1, 1), (1, 1): (-1, 0), (1, 2): (1, 3), (1, 3): (-1, 2),
            (2, 0): (1, 2), (2, 1): (-1, 3), (2, 2): (-1, 0), (2, 3): (1, 1),
            (3, 0): (1, 3), (3, 1): (1, 2), (3, 2): (-1, 1), (3, 3): (-1, 0)}
    T = np.zeros((8, 8), dtype=int)
    for g in range(8):
        for h in range(8):
            (s1, u), (s2, v) = divmod(g, 4), divmod(h, 4)
            sg, w = unit[(u, v)]
            sign = sg * (-1) ** (s1 + s2)
            T[g, h] = 4 * (0 if sign > 0 else 1) + w
    return T


def symmetric_table(n: int) -> np.ndarray:
    """S_n, permutations of (0, ..., n-1) in lexicographic order; (p q)(i) = p(q(i))."""
    perms = list(itertools.permutations(range(n)))
    index = {p: i for i, p in enumerate(perms)}
    T = np.zeros((len(perms), len(perms)), dtype=int)
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            T[i, j] = index[tuple(p[q[k]] for k in range(n))]
    return T


def product_table(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """G_1 x G_2: element (g, h) has index g |G_2| + h."""
    n1, n2 = len(t1), len(t2)
    T = t1[:, None, :, None] * n2 + t2[None, :, None, :]
    return T.reshape(n1 * n2, n1 * n2)


def check_table(T: np.ndarray) -> tuple:
    """Check that T is the multiplication table of a group; return (identity, inverses)."""
    T = np.asarray(T)
    n = len(T)
    if T.shape != (n, n) or T.min() < 0 or T.max() >= n:
        raise ValueError("the multiplication table must be an n x n array of indices 0, ..., n-1")
    ar = np.arange(n)
    if not (np.all(np.sort(T, axis=0) == ar[:, None]) and np.all(np.sort(T, axis=1) == ar[None, :])):
        raise ValueError("the multiplication table is not a Latin square (not a group)")
    ids = [e for e in range(n) if np.array_equal(T[e], ar)]
    if len(ids) != 1 or not np.array_equal(T[:, ids[0]], ar):
        raise ValueError("the multiplication table has no two-sided identity")
    e = ids[0]
    if n <= 128:
        ok = np.array_equal(T[T], T[:, T])            # (g h) k == g (h k) for all triples
    else:
        rng = np.random.default_rng(0)
        a, b, c = rng.integers(0, n, (3, 200000))
        ok = np.array_equal(T[T[a, b], c], T[a, T[b, c]])
    if not ok:
        raise ValueError("the multiplication table is not associative")
    inv = np.argmax(T == e, axis=1)
    return e, inv


# ---------------------------------------------------------------------------
# Irreducible representations from the multiplication table
# ---------------------------------------------------------------------------
def _complex_irreps(T, inv, seed=12345, tries=6):
    """One unitary matrix representative per irrep of the group with table T.

    Method: right multiplication R(h) by a random self-adjoint h = h* in C[G] commutes with
    the left regular representation, and for generic h its eigenspaces are exactly the
    irreducible left-invariant subspaces (l^2(G) = sum_rho V_rho (x) V_rho^*, and R(h) acts
    on the second factor).  Restricting left translation to an eigenspace with orthonormal
    basis B gives rho(g) = B^H L(g) B.  Equivalent irreps are recognized by their
    characters.  Returns a list of (chi, rho) with rho of shape (|G|, d, d).
    """
    n = len(T)
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    left = T[inv]                                   # left[g, k] = g^{-1} k: (L(g) B)[k] = B[g^{-1} k]
    for _ in range(tries):
        r = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        h = r + np.conj(r[inv])                     # h*(g) = conj(h(g^{-1})) = h(g)
        Rm = np.zeros((n, n), dtype=complex)
        for j in range(n):
            Rm[T[k, j], k] += h[j]                  # (x h)(k j) gets x(k) h(j)
        Rm = 0.5 * (Rm + Rm.conj().T)
        w, V = np.linalg.eigh(Rm)
        cut = np.flatnonzero(np.diff(w) > 1e-8 * max(1.0, np.abs(w).max())) + 1
        clusters = np.split(np.arange(n), cut)
        classes = []                                # [chi, rho, count]
        ok = True
        for idx in clusters:
            B = V[:, idx]
            rho = np.einsum("kd,gke->gde", B.conj(), B[left])
            chi = np.trace(rho, axis1=1, axis2=2)
            if abs(np.vdot(chi, chi).real - n) > 1e-6 * n:   # not irreducible: eigenvalues collided
                ok = False
                break
            for c in classes:
                if np.linalg.norm(c[0] - chi) < 1e-6 * np.sqrt(n):
                    c[2] += 1
                    break
            else:
                classes.append([chi, rho, 1])
        if not ok:
            continue
        dims = [c[1].shape[-1] for c in classes]
        if sum(d * d for d in dims) != n or any(c[2] != c[1].shape[-1] for c in classes):
            continue
        return [(c[0], c[1]) for c in classes]
    raise RuntimeError("could not separate the irreducible representations; is the table a group?")


def _intertwiner(rho, rng):
    """T with rho(g) T = T conj(rho(g)) for all g, scaled to be unitary (for nu != 0).
    T^T = nu T: symmetric for real type, antisymmetric for quaternionic type."""
    d = rho.shape[-1]
    for _ in range(10):
        X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        Tm = np.einsum("gij,jk,glk->il", rho, X, rho)
        s = np.sqrt(np.trace(Tm @ Tm.conj().T).real / d)
        if s > 1e-6 * len(rho):
            return Tm / s
    raise RuntimeError("no intertwiner found")


def _orth(x, basis):
    for _ in range(2):
        for b in basis:
            x = x - np.vdot(b, x) * b
    return x


def _real_form(rho, rng):
    """Real orthogonal realization of a real-type irrep: an orthonormal basis of the fixed
    points of the antiunitary theta(x) = T conj(x) (theta^2 = 1), in which rho is real."""
    d = rho.shape[-1]
    Tm = _intertwiner(rho, rng)
    basis = []
    while len(basis) < d:
        v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        x = v + Tm @ v.conj()                       # fixed by theta
        for _ in range(2):
            for b in basis:
                x = x - np.real(np.vdot(b, x)) * b  # real coefficients keep x fixed
        nx = np.linalg.norm(x)
        if nx > 1e-6:
            basis.append(x / nx)
    U = np.stack(basis, axis=1)
    out = np.einsum("ki,gkl,lj->gij", U.conj(), rho, U)
    if np.abs(out.imag).max() > 1e-8:
        raise RuntimeError("real form of a real-type irrep failed")
    return np.ascontiguousarray(out.real)


def _quaternion_form(rho, rng):
    """Interleaved quaternionic realization of a quaternionic-type irrep.

    theta(x) = T conj(x) is antiunitary with theta^2 = -1 and commutes with rho.  A
    quaternionic Gram-Schmidt gives an orthonormal basis u_1, w_1, ..., u_m, w_m with
    w_j = -theta(u_j); then T = V J V^T with J = I_m (x) [[0, 1], [-1, 0]], and
    rho'(g) = V^H rho(g) V satisfies rho' J = J conj(rho'), i.e. every 2 x 2 sub-block has
    the form [[z, w], [-conj(w), conj(z)]] (the convention of MatrixAlgebra(d, 'H'))."""
    d = rho.shape[-1]
    Tm = _intertwiner(rho, rng)
    cols = []
    while len(cols) < d:
        u = _orth(rng.standard_normal(d) + 1j * rng.standard_normal(d), cols)
        nu = np.linalg.norm(u)
        if nu < 1e-6:
            continue
        u = u / nu
        cols += [u, -Tm @ u.conj()]
    V = np.stack(cols, axis=1)
    out = np.einsum("ki,gkl,lj->gij", V.conj(), rho, V)
    J = np.kron(np.eye(d // 2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    if np.abs(out @ J - J @ out.conj()).max() > 1e-8:
        raise RuntimeError("quaternionic form of a quaternionic-type irrep failed")
    return out


def _blocks(T, inv, field):
    """[(division, d, weight, matrices (|G|, m, m))] for the structure map, sorted by type and size."""
    n = len(T)
    rng = np.random.default_rng(2024)
    irreps = _complex_irreps(T, inv)
    blocks = []
    used = set()
    for i, (chi, rho) in enumerate(irreps):
        d = rho.shape[-1]
        if field == "C":
            blocks.append(("C", d, d / n, rho))
            continue
        if i in used:
            continue
        nu = int(np.rint((chi[np.diag(T)].sum() / n).real))
        if nu == 1:
            blocks.append(("R", d, d / n, _real_form(rho, rng)))
        elif nu == 0:
            # store one of the pair (rho, conj(rho)): skip the partner
            for j, (chj, _) in enumerate(irreps):
                if j != i and np.linalg.norm(chj - np.conj(chi)) < 1e-6 * np.sqrt(n):
                    used.add(j)
            blocks.append(("C", d, 2 * d / n, rho))
        elif nu == -1:
            blocks.append(("H", d // 2, d / n, _quaternion_form(rho, rng)))
        else:
            raise RuntimeError(f"Frobenius-Schur indicator {nu} is not -1, 0 or 1")
    order = {"R": 0, "C": 1, "H": 2}
    return sorted(blocks, key=lambda b: (order[b[0]], b[3].shape[-1]))


# ---------------------------------------------------------------------------
# Shared algebra operations for group algebras
# ---------------------------------------------------------------------------
class _GroupOps:
    """mul / star / one / gamma from the multiplication table (coefficient arrays of shape
    (..., *self.shape), flattened internally)."""

    is_group_algebra = True
    is_diagonal_real = False

    def _flat(self, a):
        a = np.asarray(a)
        return a.reshape(a.shape[: a.ndim - len(self.shape)] + (self._n,))

    def _unflat(self, a):
        return a.reshape(a.shape[:-1] + self.shape)

    def mul(self, a, b):
        af, bf = self._flat(a), self._flat(b)
        T = self.cayley_table()
        out = np.zeros(np.broadcast_shapes(af.shape, bf.shape), dtype=np.result_type(af, bf))
        for h in range(self._n):
            out[..., T[h]] += af[..., h, None] * bf          # T[h] is a permutation: no repeats
        return self._unflat(out)

    def star(self, a):
        s = self._flat(a)[..., self._inv]
        return self._unflat(np.conj(s) if self.field == "C" else s)

    def one(self):
        e = np.zeros(self._n, dtype=float if self.field == "R" else complex)
        e[self._identity] = 1.0
        return e.reshape(self.shape)

    # one-axis structure constants, used when a group algebra is a factor of a TensorProduct
    def gamma(self):
        T = self.cayley_table()
        G = np.zeros((self._n,) * 3)
        i, j = np.meshgrid(np.arange(self._n), np.arange(self._n), indexing="ij")
        G[i, j, T] = 1.0
        return G

    def _star_matrix(self):
        S = np.zeros((self._n, self._n))
        S[np.arange(self._n), self._inv] = 1.0
        return S


# ---------------------------------------------------------------------------
# Abelian groups: the FFT fast path
# ---------------------------------------------------------------------------
class AbelianGroupAlgebra(_GroupOps, Algebra):
    """F[Z_{n_1} x ... x Z_{n_k}] with the k-dimensional DFT as structure map.

    Elements are arrays a(g) on the grid Z_{n_1} x ... x Z_{n_k} (k coordinate axes) with
    k-dimensional circular convolution as product: the algebra for data periodic in several
    directions at once.  One 1 x 1 block per wavevector m:
    * field 'R': the blocks with m = -m (every m_i in {0, n_i / 2}) are real; the others
      come in conjugate pairs m, -m, stored once (the smaller flat index), weight 2 / N,
      N = n_1 ... n_k; real blocks weight 1 / N.
    * field 'C': all N blocks complex, weight 1 / N.
    For k = 1 it equals TubeDFT(n).
    """

    def __init__(self, *ns: int, field: str = "R"):
        if len(ns) < 1:
            raise ValueError("AbelianGroupAlgebra needs at least one cyclic factor")
        if field not in ("R", "C"):
            raise ValueError("field must be 'R' or 'C'")
        self.shape = tuple(int(n) for n in ns)
        self.field = field
        self.is_commutative = True
        self.group_name = " x ".join(f"Z_{n}" for n in self.shape)
        self.name = f"{field}[{self.group_name}]"
        N = int(np.prod(self.shape))
        flat = np.arange(N)
        multi = np.unravel_index(flat, self.shape)
        neg = np.ravel_multi_index(tuple((-m) % n for m, n in zip(multi, self.shape)), self.shape)
        self._n = N
        self._identity = 0
        self._inv = neg
        self._real_idx = flat[neg == flat]
        self._pair_idx = flat[flat < neg]
        self._pair_neg = neg[flat < neg]

    def cayley_table(self):
        return abelian_table(self.shape)

    @property
    def _axes(self):
        return tuple(range(-len(self.shape), 0))

    def forward(self, x):
        x = np.asarray(x)
        self._check_shape(x)
        k, N = len(self.shape), self._n
        xh = np.fft.fftn(x, axes=self._axes)
        xh = xh.reshape(xh.shape[:-k] + (N,))
        if self.field == "C":
            return [Group(xh[..., :, None, None], "C", 1, np.full(N, 1.0 / N))]
        groups = [Group(np.ascontiguousarray(xh[..., self._real_idx].real)[..., None, None],
                        "R", 1, np.full(len(self._real_idx), 1.0 / N))]
        if len(self._pair_idx):
            groups.append(Group(np.ascontiguousarray(xh[..., self._pair_idx])[..., None, None],
                                "C", 1, np.full(len(self._pair_idx), 2.0 / N)))
        return groups

    def inverse(self, groups):
        N = self._n
        if self.field == "C":
            c = groups[0].data[..., 0, 0]
            return np.fft.ifftn(c.reshape(c.shape[:-1] + self.shape), axes=self._axes)
        r = np.real(groups[0].data[..., 0, 0])
        lead = r.shape[:-1]
        dtype = np.complex64 if r.dtype == np.float32 else complex
        xh = np.zeros(lead + (N,), dtype=dtype)
        xh[..., self._real_idx] = r
        if len(self._pair_idx):
            c = groups[1].data[..., 0, 0]
            xh[..., self._pair_idx] = c
            xh[..., self._pair_neg] = np.conj(c)
        return np.fft.ifftn(xh.reshape(lead + self.shape), axes=self._axes).real

    def mul(self, a, b):     # faster than the table: convolution theorem
        ax = self._axes
        p = np.fft.ifftn(np.fft.fftn(a, axes=ax) * np.fft.fftn(b, axes=ax), axes=ax)
        return p.real if self.field == "R" else p


TubeDFTnd = AbelianGroupAlgebra      # earlier name (stage vlasov2d-16), kept for compatibility


# ---------------------------------------------------------------------------
# Any finite group
# ---------------------------------------------------------------------------
class GroupAlgebra(_GroupOps, Algebra):
    """F[G] for a finite group G given by its multiplication table (see the module docstring).

    Parameters
    ----------
    table : (|G|, |G|) integer array, table[g, h] = index of g h.
    shape : coordinate shape (default (|G|,)); elements in C order of it.  Products of
            groups use one axis per factor.
    field : 'R' or 'C'.
    group_name : used in the algebra's name, e.g. 'D_4' gives 'R[D_4]'.
    """

    def __init__(self, table, shape=None, field: str = "R", group_name: str = "G"):
        if field not in ("R", "C"):
            raise ValueError("field must be 'R' or 'C'")
        T = np.asarray(table, dtype=int)
        n = len(T)
        if n > MAX_ORDER:
            raise ValueError(f"group of order {n} > {MAX_ORDER}: the numerical irreps need "
                             f"|G| x |G| eigendecompositions; for an abelian group use AbelianGroupAlgebra")
        self._identity, self._inv = check_table(T)
        self._table = T
        self._n = n
        self.shape = (n,) if shape is None else tuple(int(s) for s in shape)
        if int(np.prod(self.shape)) != n:
            raise ValueError(f"shape {self.shape} does not have |G| = {n} elements")
        self.field = field
        self.is_commutative = bool(np.array_equal(T, T.T))
        self.group_name = group_name
        self.name = f"{field}[{group_name}]"
        # blocks: (division, d, weight, matrices (n, m, m)), merged into groups of equal type and size
        self._groups = []
        for blk in _blocks(T, self._inv, field):
            div, d, w, M = blk
            if self._groups and self._groups[-1][0] == div and self._groups[-1][1] == d:
                self._groups[-1][2].append(w)
                self._groups[-1][3].append(M)
            else:
                self._groups.append([div, d, [w], [M]])
        self._mats = []    # per group: (division, d, weights (B,), Rmat (n, B*m*m), m, B)
        for div, d, ws, Ms in self._groups:
            R = np.stack(Ms, axis=1)                         # (n, B, m, m)
            B, m = R.shape[1], R.shape[-1]
            self._mats.append((div, d, np.array(ws), R.reshape(n, B * m * m), m, B))

    def cayley_table(self):
        return self._table

    # ---- constructors -------------------------------------------------------------
    @staticmethod
    def abelian(*ns, field="R"):
        """F[Z_{n_1} x ... x Z_{n_k}] on the FFT fast path (an AbelianGroupAlgebra)."""
        return AbelianGroupAlgebra(*ns, field=field)

    @classmethod
    def dihedral(cls, n, field="R"):
        """D_n, order 2n: one coordinate axis, r^k s^j at index 2k + j."""
        return cls(dihedral_table(n), field=field, group_name=f"D_{n}")

    @classmethod
    def quaternion(cls, field="R"):
        """Q_8, order 8.  R[Q_8] = R^4 (+) H: four real 1 x 1 blocks and one quaternion block."""
        return cls(quaternion_table(), field=field, group_name="Q_8")

    @classmethod
    def symmetric(cls, n, field="R"):
        """S_n, order n!, permutations in lexicographic order."""
        return cls(symmetric_table(n), field=field, group_name=f"S_{n}")

    @classmethod
    def product(cls, *algebras, field=None):
        """The group algebra of G_1 x ... x G_m from group algebras of the factors (TubeDFT,
        AbelianGroupAlgebra or GroupAlgebra).  Coordinates: the factors' axes in order.
        An AbelianGroupAlgebra if every factor is one."""
        if not algebras or not all(getattr(K, "is_group_algebra", False) for K in algebras):
            raise ValueError("GroupAlgebra.product needs group algebras")
        fields = {K.field for K in algebras}
        if field is None:
            if len(fields) != 1:
                raise ValueError("the factors have different fields")
            field = fields.pop()
        from .tubes import TubeDFT
        if all(isinstance(K, (TubeDFT, AbelianGroupAlgebra)) for K in algebras):
            return AbelianGroupAlgebra(*[n for K in algebras for n in K.shape], field=field)
        T = algebras[0].cayley_table()
        for K in algebras[1:]:
            T = product_table(T, K.cayley_table())
        shape = tuple(s for K in algebras for s in K.shape)
        name = " x ".join(K.group_name for K in algebras)
        return cls(T, shape=shape, field=field, group_name=name)

    # ---- structure map ------------------------------------------------------------
    def irrep_types(self):
        """[(division, d, number of blocks)] of the structure map, e.g. for R[Q_8]:
        [('R', 1, 4), ('H', 1, 1)]."""
        return [(div, d, B) for div, d, _, _, _, B in self._mats]

    def forward(self, x):
        x = np.asarray(x)
        self._check_shape(x)
        xf = self._flat(x)
        single = xf.dtype in (np.float32, np.complex64)
        out = []
        for div, d, ws, R, m, B in self._mats:
            Rc = R.astype(np.complex64 if np.iscomplexobj(R) else np.float32) if single else R
            data = (xf @ Rc).reshape(xf.shape[:-1] + (B, m, m))
            if div == "R" and self.field == "R":
                data = np.real(data)
            out.append(Group(data, div, d, ws))
        return out

    def inverse(self, groups):
        acc = None
        for g, (div, d, ws, R, m, B) in zip(groups, self._mats):
            data = g.data
            Rw = np.conj(R.reshape(self._n, B, m * m) * ws[None, :, None]).reshape(self._n, B * m * m)
            if data.dtype in (np.float32, np.complex64):
                Rw = Rw.astype(np.complex64 if np.iscomplexobj(Rw) else np.float32)
            c = data.reshape(data.shape[:-3] + (B * m * m,)) @ Rw.T
            acc = c if acc is None else acc + c
        if self.field == "R":
            acc = np.real(acc)
        return self._unflat(acc)
