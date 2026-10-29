"""staralg: SVDs and SVD-based tensor decompositions over finite-dimensional *-algebras."""
from .algebras import *  # noqa: F401,F403
from .algebras import __all__ as _alg_all
from .core import *  # noqa: F401,F403
from .core import __all__ as _core_all
from .metrics import compression_ratio, real_size, rel_error

__all__ = list(_alg_all) + list(_core_all) + ["rel_error", "real_size", "compression_ratio"]
__version__ = "0.7.0"
