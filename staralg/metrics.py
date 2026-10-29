"""Error and storage metrics, following Kilmer, Horesh, Avron and Newman (PNAS 2021).

* relative error      ||X - X_hat||_F / ||X||_F, always computed in coefficient form;
* real size           number of real numbers in an array (complex entries count twice);
* compression ratio   real numbers of the uncompressed data / real parameters stored.
"""
from __future__ import annotations

import numpy as np


def rel_error(X: np.ndarray, X_hat: np.ndarray) -> float:
    X = np.asarray(X)
    X_hat = np.asarray(X_hat)
    if X.shape != X_hat.shape:
        raise ValueError(f"shape mismatch: {X.shape} vs {X_hat.shape}")
    return float(np.linalg.norm((X - X_hat).ravel()) / np.linalg.norm(X.ravel()))


def real_size(X: np.ndarray) -> int:
    X = np.asarray(X)
    return int(X.size * (2 if np.iscomplexobj(X) else 1))


def compression_ratio(n_data_real: float, n_params: float) -> float:
    return float(n_data_real) / float(n_params) if n_params > 0 else float("inf")
