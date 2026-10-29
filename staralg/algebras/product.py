"""Tensor products of algebras, e.g. R[Z_64] (x) DCT_32 or C[Z_n] (x) DCT_m.

Supported:
* any number of real *_M factors (StarM, TubeDCT, WalshHadamard, ...) together with at
  most one other one-axis factor (a group algebra or a Clifford algebra).  A *_M factor
  has real 1x1 blocks, so the blocks of the product are the blocks of the other factor
  repeated once for every *_M coordinate.
* a product of group algebras only (TubeDFT, AbelianGroupAlgebra, GroupAlgebra; same
  field): F[G] (x) F[H] = F[G x H], so TensorProduct returns GroupAlgebra.product of the
  factors, e.g. TensorProduct(TubeDFT(32), TubeDFT(32)) is AbelianGroupAlgebra(32, 32).
  (Over R the blocks of a product are not products of blocks, C (x)_R C = C (+) C,
  C (x)_R H = M_2(C); the irreps of the product group give them directly.)

The coefficient axes are the factors' axes, in the order the factors are given.
Over C (when the non-*_M factor has field C) the *_M factors act complex-linearly
and their involution is complex conjugation.
"""
from __future__ import annotations

import string

import numpy as np

from .base import Algebra, Group, Trivial


class TensorProduct(Algebra):
    def __new__(cls, *factors):
        # F[G] (x) F[H] = F[G x H]: a tensor product of group algebras (same field) is the
        # group algebra of the product group, built by GroupAlgebra.product (an
        # AbelianGroupAlgebra on the FFT fast path when every factor is abelian).  The real
        # tensor product of complex blocks (C (x)_R C = C (+) C) is handled there by the
        # irreps of the product group.
        if len(factors) >= 2 and all(getattr(f, "is_group_algebra", False) for f in factors) \
                and len({f.field for f in factors}) == 1:
            from .group import GroupAlgebra
            return GroupAlgebra.product(*factors)
        return super().__new__(cls)

    def __init__(self, *factors):
        if len(factors) < 2:
            raise ValueError("TensorProduct needs at least two factors")
        for f in factors:
            if isinstance(f, Trivial) or len(f.shape) != 1:
                raise ValueError("every factor must be a one-axis algebra (not Trivial)")
        diag = [i for i, f in enumerate(factors) if getattr(f, "is_diagonal_real", False)]
        main = [i for i in range(len(factors)) if i not in diag]
        if len(main) > 1:
            raise NotImplementedError(
                "At most one factor may be a non-*_M algebra (a group algebra or a Clifford "
                "algebra); the others must be real *_M products such as TubeDCT.  (A tensor "
                "product of group algebras only is supported: it is the group algebra of the "
                "product group.)"
            )
        self.factors = tuple(factors)
        self.shape = tuple(f.shape[0] for f in factors)
        self.field = "C" if any(f.field == "C" for f in factors) else "R"
        self.is_commutative = all(f.is_commutative for f in factors)
        self.name = " (x) ".join(f.name for f in factors)
        self._diag = diag
        self._main_index = main[0] if main else None
        self._main = factors[main[0]] if main else Trivial(self.field)
        # order of the coordinate axes after reordering: *_M axes first, then the main axis
        self._order = diag + main
        # block counts of the main factor, per group
        zero = np.zeros(self._main.shape, dtype=float if self._main.field == "R" else complex)
        self._main_groups = self._main.forward(zero)

    @property
    def _S(self) -> tuple:
        return tuple(self.factors[i].shape[0] for i in self._diag)

    # ---- structure map -----------------------------------------------------
    def forward(self, x):
        x = np.asarray(x)
        self._check_shape(x)
        F = len(self.factors)
        nd = len(self._diag)
        # reorder coordinate axes: (..., *_M axes..., main axis)
        y = np.moveaxis(x, [-F + i for i in self._order], list(range(-F, 0)))
        # apply the *_M transforms on their axes
        for pos, i in enumerate(self._diag):
            ax = -F + pos
            y = np.moveaxis(self.factors[i]._transform(np.moveaxis(y, ax, -1)), -1, ax)
        groups = self._main.forward(y)  # data: (..., S_1, ..., S_k, B, m, m)
        S = int(np.prod(self._S)) if nd else 1
        out = []
        for g in groups:
            data = g.data
            lead = data.shape[: data.ndim - 3 - nd]
            data = data.reshape(lead + (S * g.n_blocks, g.m, g.m))
            out.append(Group(data, g.division, g.d, np.tile(g.weights, S)))
        return out

    def inverse(self, groups):
        F = len(self.factors)
        nd = len(self._diag)
        S = self._S
        main_groups = []
        for g, g0 in zip(groups, self._main_groups):
            data = g.data
            lead = data.shape[:-3]
            data = data.reshape(lead + S + (g0.n_blocks, g.m, g.m))
            main_groups.append(g0.with_data(data))
        y = self._main.inverse(main_groups)  # (..., S_1, ..., S_k[, n_main])
        # the *_M axes occupy positions -F, ..., -F + nd - 1 in both cases
        # (with a Trivial main factor, F == nd and there is no main axis)
        for pos, i in enumerate(self._diag):
            ax = -F + pos
            y = np.moveaxis(self.factors[i]._itransform(np.moveaxis(y, ax, -1)), -1, ax)
        return np.moveaxis(y, list(range(-F, 0)), [-F + i for i in self._order])

    # ---- algebra operations (tests) ------------------------------------------
    def mul(self, a, b):
        letters = iter(string.ascii_letters)
        ia, ib, ic, gammas = "", "", "", []
        for f in self.factors:
            i, j, k = next(letters), next(letters), next(letters)
            ia, ib, ic = ia + i, ib + j, ic + k
            gammas.append((i + j + k, f.gamma()))
        spec = f"...{ia},...{ib}," + ",".join(s for s, _ in gammas) + f"->...{ic}"
        return np.einsum(spec, a, b, *[G for _, G in gammas])

    def star(self, a):
        F = len(self.factors)
        y = a
        for pos, f in enumerate(self.factors):
            ax = -F + pos
            y = np.moveaxis(np.einsum("ij,...j->...i", f._star_matrix(), np.moveaxis(y, ax, -1)), -1, ax)
        return np.conj(y) if self.field == "C" else y

    def one(self):
        out = self.factors[0].one()
        for f in self.factors[1:]:
            out = np.multiply.outer(out, f.one())
        return out.astype(complex) if self.field == "C" else out
