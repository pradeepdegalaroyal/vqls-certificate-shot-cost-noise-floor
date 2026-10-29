"""Hardware-efficient real ansatz: ``L`` blocks of Ry rotations plus a CNOT chain, then a final Ry layer."""
from __future__ import annotations

import numpy as np


def n_params(q: int, layers: int) -> int:
    """Number of rotation angles."""
    return q * (layers + 1)


def n_cnots(q: int, layers: int) -> int:
    """Number of CNOT gates (linear chain per block)."""
    return layers * (q - 1)


def _ry(state: np.ndarray, j: int, theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    st = np.moveaxis(state, j, 0)
    out = np.empty_like(st)
    out[0] = c * st[0] - s * st[1]
    out[1] = s * st[0] + c * st[1]
    return np.moveaxis(out, 0, j)


def _cnot(state: np.ndarray, c: int, t: int) -> np.ndarray:
    out = state.copy()
    idx = [slice(None)] * state.ndim
    idx[c] = 1
    sub = state[tuple(idx)]
    t_axis = t if t < c else t - 1
    out[tuple(idx)] = np.flip(sub, axis=t_axis)
    return out


def statevector(theta: np.ndarray, q: int, layers: int) -> np.ndarray:
    """Real statevector of the ansatz applied to ``|0...0>``."""
    state = np.zeros((2,) * q)
    state[(0,) * q] = 1.0
    k = 0
    for _ in range(layers):
        for j in range(q):
            state = _ry(state, j, theta[k])
            k += 1
        for j in range(q - 1):
            state = _cnot(state, j, j + 1)
    for j in range(q):
        state = _ry(state, j, theta[k])
        k += 1
    return state.reshape(-1)
