"""Finite-dimensional *-algebras and their structure isomorphisms."""
from .base import (
    Algebra,
    Group,
    OneAxisAlgebra,
    Trivial,
    move_alg_axes_back,
    move_alg_axes_last,
)
from .directsum import DirectSum
from .clifford import Cl03, Cl04, Cl10, Cl30, Clifford, Quaternions
from .matrix import MatrixAlgebra
from .product import TensorProduct
from .tubes import RandomOrthogonal, StarM, TubeDCT, TubeDFT, WalshHadamard
from .group import AbelianGroupAlgebra, GroupAlgebra, TubeDFTnd

__all__ = [
    "Algebra", "Group", "OneAxisAlgebra", "Trivial",
    "move_alg_axes_last", "move_alg_axes_back",
    "TubeDFT", "AbelianGroupAlgebra", "GroupAlgebra", "TubeDFTnd", "StarM", "TubeDCT", "WalshHadamard", "RandomOrthogonal",
    "Clifford", "Cl10", "Cl03", "Cl04", "Cl30", "Quaternions", "MatrixAlgebra",
    "TensorProduct", "DirectSum",
]
