"""Shot cost of certifying a VQLS solution with the condition-number bound, with and without noise.

Certificate: the VQLS paper proves ``C >= eps^2 / kappa^2`` where ``eps`` is the trace distance to the
exact solution, so ``eps <= kappa * sqrt(C)``.  With a finite-shot estimate ``C_hat`` of standard
deviation ``sigma`` we issue the one-sided 95 % certificate ``eps_cert = kappa * sqrt(C_hat + 1.645 sigma)``.

Noise model (explicit, deliberately simple): every CNOT mixes in a depolarising fraction ``p`` globally, so a
circuit with ``g`` CNOTs scales every non-identity expectation value by ``(1 - p)^g``.  Zero-noise
extrapolation (ZNE) folds the circuit to noise factors ``lambda = 1, 3, 5`` and applies the quadratic
Richardson weights ``gamma``; extrapolation is applied term by term before the cost is assembled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ansatz import n_cnots
from .cost import TermSet, beta_delta

Z95 = 1.645
LAMBDAS = np.array([1.0, 3.0, 5.0])
GAMMAS = np.array([15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0])  # Lagrange weights for extrapolating to lambda = 0


def gate_counts(ts: TermSet, q: int, layers: int) -> tuple[np.ndarray, np.ndarray]:
    """CNOT counts per circuit for (Hadamard-test terms, Pauli-expectation terms).

    Assumption: all-to-all ancilla, Toffoli = 6 CNOT, controlled-Ry = 2 CNOT, no routing overhead.
    """
    g_a = n_cnots(q, layers)
    g_h = 6 * g_a + 2 * q * (layers + 1) + 2 * q + ts.w_h
    g_p = np.full(len(ts.d_p), g_a)
    return g_h.astype(float), g_p.astype(float)


@dataclass
class Analysis:
    """Moments of the finite-shot cost estimator for one (state, noise, mitigation) setting."""

    c_mean: float        # mean of the estimator (includes any residual noise bias)
    v_uniform: float     # Var * S for equal shots per circuit setting
    v_neyman: float      # Var * S for variance-optimal (Neyman) allocation
    n_items: int         # number of distinct circuit settings (terms x noise factors)
    gates_per_shot: float  # mean CNOT count per shot (uniform allocation)


def analyse(mu_h: np.ndarray, mu_p: np.ndarray, ts: TermSet, q: int, layers: int,
            p: float, zne: bool) -> Analysis:
    """Delta-method moments of the estimator ``C_hat = 1 - beta_hat^2 / delta_hat``."""
    g_h, g_p = gate_counts(ts, q, layers)
    mu = np.concatenate([mu_h, mu_p])
    g = np.concatenate([g_h, g_p])
    lam = LAMBDAS if zne else np.array([1.0])
    gam = GAMMAS if zne else np.array([1.0])
    fac = (1.0 - p) ** np.outer(g, lam)          # (terms, lambdas)
    meas = fac * mu[:, None]                     # noisy means actually measured
    mu_eff = meas @ gam                          # mitigated (or raw) term means
    nh = len(mu_h)
    beta, delta = beta_delta(mu_eff[:nh], mu_eff[nh:], ts)
    sens = np.concatenate([-2.0 * beta / delta * ts.c_h, beta ** 2 / delta ** 2 * ts.d_p])
    w = np.abs(sens[:, None] * gam[None, :])
    v = np.clip(1.0 - meas ** 2, 1e-12, None)
    return Analysis(
        c_mean=1.0 - beta ** 2 / delta,
        v_uniform=float(w.size * np.sum(w ** 2 * v)),
        v_neyman=float(np.sum(w * np.sqrt(v)) ** 2),
        n_items=int(w.size),
        gates_per_shot=float(np.mean(g)) * float(np.mean(lam)),
    )


def sigma(an: Analysis, shots: float, alloc: str) -> float:
    """Standard deviation of ``C_hat`` for ``shots`` total shots."""
    v = an.v_uniform if alloc == "uniform" else an.v_neyman
    return float(np.sqrt(v / shots))


def certified_eps(an: Analysis, shots: float, alloc: str, kappa: float) -> float:
    """Median 95 % certified precision ``kappa * sqrt(C_mean + 1.645 sigma)``."""
    return float(kappa * np.sqrt(max(an.c_mean + Z95 * sigma(an, shots, alloc), 0.0)))


def shots_to_certify(an: Analysis, alloc: str, target_c: float) -> float:
    """Total shots so the 95 % upper bound is issued below ``target_c`` with 95 % probability.

    Requires ``c_mean + 2 * 1.645 * sigma <= target_c``; returns ``inf`` if the noise bias alone
    already exceeds the target (a noise floor no number of shots can remove).
    """
    gap = target_c - an.c_mean
    if gap <= 0:
        return float("inf")
    v = an.v_uniform if alloc == "uniform" else an.v_neyman
    return float((2.0 * Z95) ** 2 * v / gap ** 2)


def monte_carlo(mu_h: np.ndarray, mu_p: np.ndarray, ts: TermSet, q: int, layers: int, p: float,
                zne: bool, shots_per_item: int, c_ideal: float, target_c: float,
                reps: int, rng: np.random.Generator) -> dict:
    """Binomial shot simulation (uniform allocation) checking the delta-method certificate.

    Returns the coverage ``P[C_ub >= C_ideal]`` of the one-sided bound and the fraction of repeats
    that issue a certificate at ``target_c``.
    """
    g_h, g_p = gate_counts(ts, q, layers)
    mu = np.concatenate([mu_h, mu_p])
    g = np.concatenate([g_h, g_p])
    lam = LAMBDAS if zne else np.array([1.0])
    gam = GAMMAS if zne else np.array([1.0])
    meas = (1.0 - p) ** np.outer(g, lam) * mu[:, None]
    k = rng.binomial(shots_per_item, (1.0 + meas) / 2.0, size=(reps,) + meas.shape)
    m_hat = 2.0 * k / shots_per_item - 1.0                       # (reps, terms, lambdas)
    mu_hat = m_hat @ gam                                         # (reps, terms)
    nh = len(mu_h)
    beta = mu_hat[:, :nh] @ ts.c_h
    delta = ts.d0 + mu_hat[:, nh:] @ ts.d_p
    c_hat = 1.0 - beta ** 2 / delta
    sens_h = -2.0 * beta[:, None] / delta[:, None] * ts.c_h[None, :]
    sens_p = (beta ** 2 / delta ** 2)[:, None] * ts.d_p[None, :]
    sens = np.concatenate([sens_h, sens_p], axis=1)              # (reps, terms)
    var_item = np.clip(1.0 - m_hat ** 2, 1e-12, None) / shots_per_item
    var = np.sum((sens[:, :, None] * gam[None, None, :]) ** 2 * var_item, axis=(1, 2))
    ub = c_hat + Z95 * np.sqrt(var)
    return {"coverage": float(np.mean(ub >= c_ideal)),
            "issue_rate": float(np.mean(ub <= target_c)),
            "mean_c_hat": float(np.mean(c_hat)),
            "sd_c_hat": float(np.std(c_hat)),
            "total_shots": int(shots_per_item * meas.size)}
