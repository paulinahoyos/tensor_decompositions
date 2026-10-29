"""Matrices over K in coefficient form, and their block form.

A matrix A in K^{m x n} is an array of shape (m, n, *K.shape).  Under the
structure isomorphism, block rho of A is the (m s) x (n s) matrix whose (i, j)
sub-block of size s x s is pi_rho(A_ij), where s is the stored block size
(s = d, or 2d for quaternion blocks):

    blockmat[(i, a), (j, c)] = pi_rho(A_ij)[a, c],     rows/cols ordered (i, a).
"""
from __future__ import annotations

import numpy as np

from ..algebras.base import Algebra, Group


def matrix_blocks(groups: list[Group]) -> list[np.ndarray]:
    """Groups with data (m, n, B, s, s)  ->  list of arrays (B, m s, n s)."""
    out = []
    for g in groups:
        m, n, B, s, _ = g.data.shape
        out.append(g.data.transpose(2, 0, 3, 1, 4).reshape(B, m * s, n * s))
    return out


def blocks_to_groups(mats: list[np.ndarray], templates: list[Group], m: int, n: int) -> list[Group]:
    """Inverse of ``matrix_blocks`` for an m x n matrix over K."""
    out = []
    for M, g in zip(mats, templates):
        s = g.m
        B = M.shape[0]
        data = M.reshape(B, m, s, n, s).transpose(1, 3, 0, 2, 4)
        if g.division == "R":
            data = np.real(data)
        out.append(g.with_data(data))
    return out


def to_blocks(A: np.ndarray, K: Algebra):
    """Matrix over K (coefficient form)  ->  (block matrices, templates)."""
    if A.ndim != 2 + K.n_axes:
        raise ValueError(f"expected a matrix over {K.name} of shape (m, n, *{K.shape}), got {A.shape}")
    groups = K.forward(A)
    return matrix_blocks(groups), groups


def from_blocks(mats: list[np.ndarray], templates: list[Group], K: Algebra, m: int, n: int) -> np.ndarray:
    return K.inverse(blocks_to_groups(mats, templates, m, n))


def matmul_K(A: np.ndarray, B: np.ndarray, K: Algebra) -> np.ndarray:
    """Product of matrices over K, (AB)_ij = sum_k A_ik B_kj with K's multiplication."""
    m, p = A.shape[:2]
    p2, n = B.shape[:2]
    if p != p2:
        raise ValueError(f"inner dimensions differ: {A.shape[:2]} and {B.shape[:2]}")
    MA, gA = to_blocks(A, K)
    MB, _ = to_blocks(B, K)
    return from_blocks([a @ b for a, b in zip(MA, MB)], gA, K, m, n)


def conj_transpose_K(A: np.ndarray, K: Algebra) -> np.ndarray:
    """A^H with (A^H)_ij = (A_ji)^*."""
    return K.star(np.swapaxes(A, 0, 1))


def eye_K(n: int, K: Algebra) -> np.ndarray:
    one = np.asarray(K.one())
    I = np.zeros((n, n) + tuple(K.shape), dtype=one.dtype)
    for i in range(n):
        I[i, i] = one
    return I
