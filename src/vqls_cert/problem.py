"""Linear-system instances: a shifted transverse-field Ising operator written as a Pauli sum.

The matrix is ``A = sum_l c_l P_l`` with every ``P_l`` a tensor product of ``I``, ``X`` and ``Z``
(so ``A`` is real symmetric), shifted and scaled so that the spectral norm is 1 and the condition
number equals a requested value ``kappa`` exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import numpy as np

_I = np.eye(2)
_X = np.array([[0.0, 1.0], [1.0, 0.0]])
_Z = np.diag([1.0, -1.0])
_OPS = {"I": _I, "X": _X, "Z": _Z}


def pauli_matrix(label: str) -> np.ndarray:
    """Dense matrix of a Pauli string such as ``"XIZ"`` (qubit 0 is the leftmost factor)."""
    return reduce(np.kron, [_OPS[c] for c in label])


def weight(label: str) -> int:
    """Number of non-identity factors of a Pauli string."""
    return sum(c != "I" for c in label)


@dataclass
class Instance:
    """One linear system ``A x = b`` with a Pauli-sum ``A`` and a product-state ``b``."""

    q: int
    kappa: float
    seed: int
    labels: list[str]
    coefs: np.ndarray
    A: np.ndarray
    b_angles: np.ndarray
    b: np.ndarray
    x_star: np.ndarray

    @property
    def n(self) -> int:
        return 2 ** self.q


def product_state(angles: np.ndarray) -> np.ndarray:
    """Real product state ``Ry(angle_j)|0>`` on each qubit."""
    vecs = [np.array([np.cos(a / 2), np.sin(a / 2)]) for a in angles]
    return reduce(np.kron, vecs)


def make_instance(q: int, kappa: float, seed: int) -> Instance:
    """Build a random Ising-type instance with exact condition number ``kappa``."""
    rng = np.random.default_rng(10_000 + seed)
    labels: list[str] = []
    coefs: list[float] = []
    for j in range(q):
        labels.append("".join("X" if k == j else "I" for k in range(q)))
        coefs.append(rng.uniform(0.5, 1.5))
    for j in range(q - 1):
        labels.append("".join("Z" if k in (j, j + 1) else "I" for k in range(q)))
        coefs.append(rng.uniform(0.3, 1.0))
    h = sum(c * pauli_matrix(l) for c, l in zip(coefs, labels))
    ev = np.linalg.eigvalsh(h)
    shift = (ev[-1] - kappa * ev[0]) / (kappa - 1.0)
    scale = ev[-1] + shift
    labels = ["I" * q] + labels
    coefs_arr = np.array([shift] + coefs) / scale
    a = sum(c * pauli_matrix(l) for c, l in zip(coefs_arr, labels))
    b_angles = rng.uniform(0.3, 2.8, size=q)
    b = product_state(b_angles)
    x = np.linalg.solve(a, b)
    x /= np.linalg.norm(x)
    return Instance(q, float(kappa), seed, labels, coefs_arr, a, b_angles, b, x)
