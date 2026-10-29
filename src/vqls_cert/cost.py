"""Global VQLS cost ``C = 1 - |<b|A|x>|^2 / <x|A^2|x>`` and its measurable term decomposition.

``beta = <b|A|x> = sum_l c_l <b|P_l|x>`` is a sum of Hadamard-test overlaps.
``delta = <x|A^2|x> = sum_l c_l^2 + 2 sum_{l<m, commuting} c_l c_m <x|P_l P_m|x>`` is a sum of Pauli
expectation values (anticommuting pairs cancel).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .ansatz import statevector
from .problem import Instance, pauli_matrix


@dataclass
class TermSet:
    """Measurable decomposition of the cost for one instance."""

    c_h: np.ndarray            # coefficients of the Hadamard-test terms (one per Pauli of A)
    P_h: list[np.ndarray]      # dense Pauli matrices of those terms
    w_h: np.ndarray            # Pauli weights (for gate counting)
    d0: float                  # constant part of delta
    d_p: np.ndarray            # coefficients of the Pauli-expectation terms of delta
    M_p: list[np.ndarray]      # dense (signed) Pauli matrices of those terms


def build_terms(inst: Instance) -> TermSet:
    """Decompose beta and delta into measurable terms (duplicate Pauli products merged)."""
    mats = [pauli_matrix(l) for l in inst.labels]
    w = np.array([sum(ch != "I" for ch in l) for l in inst.labels])
    d0 = float(np.sum(inst.coefs ** 2))
    merged: dict[bytes, list] = {}
    L = len(mats)
    for l in range(L):
        for m in range(l + 1, L):
            comm = mats[l] @ mats[m] - mats[m] @ mats[l]
            if np.abs(comm).max() > 1e-12:
                continue
            M = mats[l] @ mats[m]
            key = np.round(M, 8).tobytes()
            coef = 2.0 * inst.coefs[l] * inst.coefs[m]
            if key in merged:
                merged[key][0] += coef
            else:
                merged[key] = [coef, M]
    d_p = np.array([v[0] for v in merged.values()])
    M_p = [v[1] for v in merged.values()]
    return TermSet(inst.coefs.copy(), mats, w, d0, d_p, M_p)


def term_means(x: np.ndarray, inst: Instance, ts: TermSet) -> tuple[np.ndarray, np.ndarray]:
    """Ideal means ``(mu_h, mu_p)`` of the Hadamard-test and Pauli-expectation terms."""
    mu_h = np.array([inst.b @ (P @ x) for P in ts.P_h])
    mu_p = np.array([x @ (M @ x) for M in ts.M_p])
    return mu_h, mu_p


def beta_delta(mu_h: np.ndarray, mu_p: np.ndarray, ts: TermSet) -> tuple[float, float]:
    """Combine term means into ``(beta, delta)``."""
    return float(ts.c_h @ mu_h), float(ts.d0 + ts.d_p @ mu_p)


def exact_cost(x: np.ndarray, inst: Instance) -> float:
    """Exact global cost for a normalised real state ``x``."""
    ax = inst.A @ x
    return float(1.0 - (inst.b @ ax) ** 2 / (ax @ ax))


def trace_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Trace distance between two real pure states."""
    return float(np.sqrt(max(0.0, 1.0 - float(x @ y) ** 2)))


def _beta_delta_of(theta: np.ndarray, inst: Instance, layers: int) -> tuple[float, float]:
    x = statevector(theta, inst.q, layers)
    ax = inst.A @ x
    return float(inst.b @ ax), float(ax @ ax)


def cost_and_grad(theta: np.ndarray, inst: Instance, layers: int) -> tuple[float, np.ndarray]:
    """Cost and exact gradient (parameter-shift rules on ``beta`` and ``delta``, then the chain rule).

    ``beta`` is linear in the state amplitudes (half-angle frequency), so
    ``d beta / d theta = beta(theta + pi) / 2``; ``delta`` is quadratic, so the standard
    ``pi/2`` shift applies.
    """
    beta, delta = _beta_delta_of(theta, inst, layers)
    g = np.zeros_like(theta)
    for j in range(len(theta)):
        tp = theta.copy(); tp[j] += np.pi
        db = 0.5 * _beta_delta_of(tp, inst, layers)[0]
        tp = theta.copy(); tp[j] += np.pi / 2
        tm = theta.copy(); tm[j] -= np.pi / 2
        dd = 0.5 * (_beta_delta_of(tp, inst, layers)[1] - _beta_delta_of(tm, inst, layers)[1])
        g[j] = -2.0 * beta / delta * db + beta ** 2 / delta ** 2 * dd
    return 1.0 - beta ** 2 / delta, g


def train(inst: Instance, layers: int, seed: int, maxiter: int = 400) -> dict:
    """Minimise the exact cost with L-BFGS from a small random start; record every iterate."""
    rng = np.random.default_rng(seed)
    n = inst.q * (layers + 1)
    theta0 = rng.normal(0.0, 0.3, size=n)
    history: list[np.ndarray] = [theta0.copy()]
    res = minimize(cost_and_grad, theta0, args=(inst, layers), jac=True, method="L-BFGS-B",
                   callback=lambda t: history.append(np.array(t)),
                   options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-12})
    return {"theta": res.x, "cost": float(res.fun), "history": history, "nit": int(res.nit)}
