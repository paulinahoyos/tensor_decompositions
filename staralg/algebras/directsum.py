"""Direct sums of algebras, K = K_1 (+) K_2 (+) ... (+) K_m.

Every finite-dimensional *-algebra is semisimple, so a non-simple one is a direct sum of
simple ones; this class builds such sums from algebras of this package with one
coordinate axis (or none, like Trivial), all over the same field.

Coefficients.  One axis: the coefficient vectors of the summands, concatenated in order
(a Trivial summand contributes one coordinate).  Multiplication, the involution and the
unit act summand by summand, so the summands annihilate each other.

Structure map.  The blocks of K are the blocks of the summands, listed summand by
summand (groups are not merged, so blocks of the same size in different summands
form separate groups).  The weights are those of the summands, because the
Euclidean norm of the concatenated coefficients is the sum of the summands' norms.

Examples:
    DirectSum(MatrixAlgebra(2, "C"), MatrixAlgebra(2, "C"))   # s and p polarization, M_2(C) (+) M_2(C)
    DirectSum(Trivial("C"), Trivial("C"))                     # its center, C (+) C
"""
from __future__ import annotations

import numpy as np

from .base import Algebra, Group, Trivial


class DirectSum(Algebra):
    is_diagonal_real = False

    def __init__(self, *summands):
        if len(summands) < 2:
            raise ValueError("DirectSum needs at least two summands")
        fields = {s.field for s in summands}
        if len(fields) != 1:
            raise ValueError("all summands must have the same field")
        for s in summands:
            if len(s.shape) > 1:
                raise ValueError("summands must have at most one coordinate axis")
        self.summands = tuple(summands)
        self.field = fields.pop()
        self._sizes = [s.dim for s in summands]
        self._offsets = np.concatenate([[0], np.cumsum(self._sizes)]).astype(int)
        self.shape = (int(self._offsets[-1]),)
        self.is_commutative = all(s.is_commutative for s in summands)
        self.name = " (+) ".join(s.name for s in summands)
        zero = np.zeros(self.shape, dtype=float if self.field == "R" else complex)
        self._ngroups = [len(s.forward(self._part(zero, i))) for i, s in enumerate(summands)]

    def _part(self, x, i):
        a, b = self._offsets[i], self._offsets[i + 1]
        p = x[..., a:b]
        return p[..., 0] if isinstance(self.summands[i], Trivial) else p

    def _join(self, parts):
        parts = [p[..., None] if isinstance(s, Trivial) else p for p, s in zip(parts, self.summands)]
        return np.concatenate(parts, axis=-1)

    # ---- structure map -----------------------------------------------------
    def forward(self, x):
        x = np.asarray(x)
        self._check_shape(x)
        out: list[Group] = []
        for i, s in enumerate(self.summands):
            out += s.forward(self._part(x, i))
        return out

    def inverse(self, groups):
        parts, k = [], 0
        for s, ng in zip(self.summands, self._ngroups):
            parts.append(s.inverse(groups[k:k + ng]))
            k += ng
        dtype = np.result_type(*parts)
        return self._join([np.asarray(p, dtype=dtype) for p in parts])

    # ---- algebra operations (tests) ------------------------------------------
    def mul(self, a, b):
        return self._join([s.mul(self._part(a, i), self._part(b, i)) for i, s in enumerate(self.summands)])

    def star(self, a):
        return self._join([s.star(self._part(a, i)) for i, s in enumerate(self.summands)])

    def one(self):
        return self._join([np.asarray(s.one()) for s in self.summands])
