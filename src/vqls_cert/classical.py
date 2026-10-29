"""Classical baseline: conjugate gradient on the sparse system, counted in floating-point operations."""
from __future__ import annotations

from functools import reduce

import numpy as np
import scipy.sparse as sp

from .problem import Instance

_SP = {"I": sp.identity(2, format="csr"),
       "X": sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]])),
       "Z": sp.csr_matrix(np.diag([1.0, -1.0]))}


def sparse_matrix(inst: Instance) -> sp.csr_matrix:
    """Sparse ``A`` built from its Pauli terms."""
    mats = [c * reduce(sp.kron, [_SP[ch] for ch in l], sp.identity(1)) for c, l in zip(inst.coefs, inst.labels)]
    return sum(mats[1:], mats[0]).tocsr()


def cg_flops_to_precision(inst: Instance, eps_target: float, max_iter: int = 500) -> dict:
    """Run CG from ``x = 0`` until the normalised iterate is within trace distance ``eps_target``.

    Flops per iteration: ``2 nnz`` for the mat-vec plus ``10 n`` for the dot products and axpys.
    """
    A = sparse_matrix(inst)
    b = inst.b
    x = np.zeros(inst.n)
    r = b.copy()
    d = r.copy()
    rs = r @ r
    per_iter = 2 * A.nnz + 10 * inst.n
    eps = 1.0
    it = 0
    for it in range(1, max_iter + 1):
        Ad = A @ d
        alpha = rs / (d @ Ad)
        x = x + alpha * d
        r = r - alpha * Ad
        rs_new = r @ r
        xn = x / np.linalg.norm(x)
        eps = float(np.sqrt(max(0.0, 1.0 - float(xn @ inst.x_star) ** 2)))
        if eps <= eps_target or rs_new < 1e-30:
            break
        d = r + (rs_new / rs) * d
        rs = rs_new
    return {"iterations": it, "flops": float(it * per_iter), "eps": eps, "nnz": int(A.nnz)}
