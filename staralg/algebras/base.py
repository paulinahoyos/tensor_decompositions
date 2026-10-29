"""Base classes for finite-dimensional *-algebras and their block (transform-domain) form.

Conventions
-----------
* Coefficient form.  An element of K is stored as an array whose trailing axes
  hold its coordinates in a fixed basis of K: shape ``(..., *K.shape)``.  Leading
  axes are batch axes (e.g. the entries of a matrix or tensor over K).

* Block form.  The structure isomorphism

      pi : K  -->  M_{d_1}(D_1) (+) ... (+) M_{d_l}(D_l),     D_rho in {R, C, H},

  is returned by ``K.forward(x)`` as a list of ``Group`` objects.  A group stacks
  blocks of the same size and type: ``data`` has shape ``(..., B, m, m)``.
  Quaternion blocks are stored in their complex 2x2 realization

      a + b i + c j + d k   |-->   [[a + b i,  c + d i],
                                    [-c + d i, a - b i]],

  so a block in M_d(H) is an m x m complex matrix with m = 2d.

* Weights.  Each block carries a weight c_rho > 0 such that

      ||x||^2 = sum_rho c_rho ||pi_rho(x)||_F^2 ,

  where ||x|| is the Euclidean norm of the coefficient vector.  The weights absorb
  unnormalized transforms (the DFT), conjugate pairs of blocks that are stored
  only once, and the factor 2 of the complex realization of H.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIVISIONS = ("R", "C", "H")


@dataclass
class Group:
    """Blocks of the same size and type, stacked along axis -3.

    Attributes
    ----------
    data : ndarray, shape (..., B, m, m)
        Real dtype for division 'R', complex dtype for 'C' and 'H'.
    division : {'R', 'C', 'H'}
        The division algebra D_rho of every block in the group.
    d : int
        Size of each block as a matrix over D_rho (m = d, or m = 2d for 'H').
    weights : ndarray, shape (B,)
        The weights c_rho of the blocks (see module docstring).
    """

    data: np.ndarray
    division: str
    d: int
    weights: np.ndarray

    def __post_init__(self):
        if self.division not in DIVISIONS:
            raise ValueError(f"division must be one of {DIVISIONS}, got {self.division!r}")
        self.weights = np.asarray(self.weights, dtype=float)

    @property
    def m(self) -> int:
        """Size of the stored (realized) square blocks."""
        return self.data.shape[-1]

    @property
    def n_blocks(self) -> int:
        return self.data.shape[-3]

    @property
    def real_params_per_entry(self) -> int:
        """Real parameters carried by one entry of a stored m x m block.

        R: 1.  C: 2.  H: 1, because a d x d quaternion matrix has 4 d^2 real
        parameters and its complex realization has (2d)^2 entries.
        """
        return {"R": 1, "C": 2, "H": 1}[self.division]

    def with_data(self, data: np.ndarray) -> "Group":
        """Same block structure, new data (used by the decomposition code)."""
        return Group(data, self.division, self.d, self.weights)


class Algebra:
    """Interface of a finite-dimensional *-algebra K over F in {R, C}.

    Subclasses provide ``forward`` and ``inverse`` (the structure isomorphism and
    its inverse) and, for the tests, ``mul``, ``star`` and ``one``.
    """

    shape: tuple = ()
    field: str = "R"
    is_commutative: bool = True
    name: str = "K"

    # ---- sizes -----------------------------------------------------------
    @property
    def dim(self) -> int:
        """Dimension of K over its field F."""
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def real_dim(self) -> int:
        """Dimension of K over R."""
        return self.dim * (2 if self.field == "C" else 1)

    @property
    def n_axes(self) -> int:
        return len(self.shape)

    # ---- structure map -----------------------------------------------------
    def forward(self, x: np.ndarray) -> list[Group]:  # pragma: no cover - abstract
        raise NotImplementedError

    def inverse(self, groups: list[Group]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- algebra operations on coefficient arrays (used by the tests) -----
    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def star(self, a: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def one(self) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def inner(self, a: np.ndarray, b: np.ndarray):
        """<a, b> = sum conj(a) b over the coordinate axes (Euclidean in coefficients)."""
        axes = tuple(range(-self.n_axes, 0))
        v = np.conj(a) * b
        if axes:
            v = v.sum(axis=axes)
        return np.real(v) if self.field == "R" else v

    def norm2(self, a: np.ndarray):
        return np.real(self.inner(a, a))

    # ---- helpers -------------------------------------------------------------
    def random(self, rng: np.random.Generator, batch: tuple = ()) -> np.ndarray:
        """Random element(s) with standard normal coefficients."""
        shape = tuple(batch) + tuple(self.shape)
        x = rng.standard_normal(shape)
        if self.field == "C":
            x = x + 1j * rng.standard_normal(shape)
        return x

    def _check_shape(self, x: np.ndarray) -> None:
        k = self.n_axes
        if k and tuple(x.shape[-k:]) != tuple(self.shape):
            raise ValueError(
                f"{self.name}: expected trailing coordinate axes of shape {self.shape}, "
                f"got array of shape {x.shape}"
            )

    def describe(self, return_str: bool = False):
        """Human-readable list of the blocks of the structure isomorphism."""
        groups = self.forward(np.zeros(self.shape, dtype=float if self.field == "R" else complex))
        lines = [f"{self.name}: field {self.field}, dim_F = {self.dim}, "
                 f"{'commutative' if self.is_commutative else 'noncommutative'}"]
        for g in groups:
            lines.append(f"  {g.n_blocks} block(s) M_{g.d}({g.division})  "
                         f"[stored as {g.m}x{g.m}, weights {np.unique(np.round(g.weights, 12))}]")
        text = "\n".join(lines)
        if return_str:
            return text
        print(text)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class OneAxisAlgebra(Algebra):
    """Algebra whose coordinates fill a single axis of length n.

    Subclasses define ``_gamma()`` (structure constants, e_i e_j = sum_k G[i,j,k] e_k),
    ``_star_matrix()`` (S with star(x) = S x, followed by complex conjugation when the
    field is C) and ``one()``.
    """

    _gamma_cache = None

    @property
    def n(self) -> int:
        return self.shape[0]

    def _gamma(self) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _star_matrix(self) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def gamma(self) -> np.ndarray:
        if self._gamma_cache is None:
            self._gamma_cache = self._gamma()
        return self._gamma_cache

    def mul(self, a, b):
        return np.einsum("...i,...j,ijk->...k", a, b, self.gamma())

    def star(self, a):
        s = np.einsum("ij,...j->...i", self._star_matrix(), a)
        return np.conj(s) if self.field == "C" else s


class Trivial(Algebra):
    """K = F itself (the classical case): no algebra axes, one 1x1 block."""

    def __init__(self, field: str = "R"):
        if field not in ("R", "C"):
            raise ValueError("field must be 'R' or 'C'")
        self.field = field
        self.shape = ()
        self.is_commutative = True
        self.name = field

    def forward(self, x):
        x = np.asarray(x)
        return [Group(x[..., None, None, None], self.field, 1, np.ones(1))]

    def inverse(self, groups):
        return groups[0].data[..., 0, 0, 0]

    def mul(self, a, b):
        return a * b

    def star(self, a):
        return np.conj(a)

    def one(self):
        return np.array(1.0 if self.field == "R" else 1.0 + 0j)


# ---------------------------------------------------------------------------
# Moving the algebra axes of a data array
# ---------------------------------------------------------------------------
def _normalize_axes(alg_axes, ndim: int) -> list[int]:
    if alg_axes is None:
        return []
    if isinstance(alg_axes, (int, np.integer)):
        alg_axes = [alg_axes]
    axes = [int(a) % ndim for a in alg_axes]
    if len(set(axes)) != len(axes):
        raise ValueError(f"repeated axis in alg_axes={alg_axes}")
    return axes


def move_alg_axes_last(X: np.ndarray, alg_axes) -> np.ndarray:
    """Move the axes listed in ``alg_axes`` (in that order) to the end of X."""
    axes = _normalize_axes(alg_axes, X.ndim)
    if not axes:
        return X
    return np.moveaxis(X, axes, list(range(X.ndim - len(axes), X.ndim)))


def move_alg_axes_back(Y: np.ndarray, alg_axes, ndim: int) -> np.ndarray:
    """Inverse of ``move_alg_axes_last``; ``ndim`` is the number of axes of the original array."""
    axes = _normalize_axes(alg_axes, ndim)
    if not axes:
        return Y
    return np.moveaxis(Y, list(range(Y.ndim - len(axes), Y.ndim)), axes)
