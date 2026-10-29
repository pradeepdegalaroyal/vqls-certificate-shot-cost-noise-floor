"""Experiment pipeline: train VQLS states, then measure the shot and noise cost of certifying them."""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ansatz import n_cnots, statevector
from .certify import (Analysis, analyse, certified_eps, monte_carlo, shots_to_certify)
from .classical import cg_flops_to_precision
from .cost import build_terms, exact_cost, term_means, train, trace_distance
from .problem import make_instance

P_GRID = [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
EPS_TARGETS = [0.1, 0.2, 0.4, 0.8]


@dataclass
class Config:
    """Sweep definition (the ``quick`` preset is a smoke test, the default is the full study)."""

    kappas: list[float] = field(default_factory=lambda: [4.0, 8.0, 16.0, 32.0])
    qubit_grid: list[int] = field(default_factory=lambda: [3, 4, 5, 6])
    depth_grid: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    base_q: int = 4
    base_kappa: float = 8.0
    base_layers: int = 3
    seeds: int = 5
    maxiter: int = 150
    p_grid: list[float] = field(default_factory=lambda: list(P_GRID))
    mc_reps: int = 4000

    @classmethod
    def quick(cls) -> "Config":
        return cls(kappas=[4.0, 8.0], qubit_grid=[3], depth_grid=[1, 2], base_q=3, base_layers=2,
                   seeds=2, maxiter=25, p_grid=[0.0, 1e-5, 1e-4], mc_reps=300)


def jobs(cfg: Config) -> list[tuple[int, float, int, int]]:
    """All distinct (q, kappa, layers, seed) training jobs."""
    out: set[tuple[int, float, int, int]] = set()
    for s in range(cfg.seeds):
        for k in cfg.kappas:
            out.add((cfg.base_q, k, cfg.base_layers, s))
        for q in cfg.qubit_grid:
            out.add((q, cfg.base_kappa, cfg.base_layers, s))
        for L in cfg.depth_grid:
            out.add((cfg.base_q, cfg.base_kappa, L, s))
    return sorted(out)


def run_job(args: tuple) -> dict:
    """Train one instance and compute everything downstream of the trained state."""
    q, kappa, layers, seed, maxiter, p_grid = args
    inst = make_instance(q, kappa, seed)
    t0 = time.time()
    res = train(inst, layers, seed, maxiter=maxiter)
    train_s = time.time() - t0
    x = statevector(res["theta"], q, layers)
    c_star = max(exact_cost(x, inst), 0.0)
    eps_true = trace_distance(x, inst.x_star)
    traj = []
    for it, th in enumerate(res["history"]):
        xi = statevector(th, q, layers)
        ci = exact_cost(xi, inst)
        traj.append({"q": q, "kappa": kappa, "layers": layers, "seed": seed, "iter": it, "cost": ci,
                     "eps_true": trace_distance(xi, inst.x_star), "eps_bound": kappa * np.sqrt(max(ci, 0.0))})
    ts = build_terms(inst)
    mu_h, mu_p = term_means(x, inst, ts)
    rows = []
    for p in p_grid:
        for zne in (False, True):
            an = analyse(mu_h, mu_p, ts, q, layers, p, zne)
            rows.append({"q": q, "kappa": kappa, "layers": layers, "seed": seed, "p": p, "zne": zne,
                         "c_mean": an.c_mean, "v_uniform": an.v_uniform, "v_neyman": an.v_neyman,
                         "n_items": an.n_items, "gates_per_shot": an.gates_per_shot})
    cg = cg_flops_to_precision(inst, eps_true)
    final = {"q": q, "kappa": kappa, "layers": layers, "seed": seed, "cost": c_star, "eps_true": eps_true,
             "eps_bound": kappa * np.sqrt(c_star), "nit": res["nit"], "train_s": train_s,
             "n_cnots": n_cnots(q, layers), "n_terms_h": len(ts.c_h), "n_terms_p": len(ts.d_p),
             "cg_iters": cg["iterations"], "cg_flops": cg["flops"], "cg_eps": cg["eps"]}
    return {"traj": traj, "rows": rows, "final": final, "mu_h": mu_h, "mu_p": mu_p,
            "inst": (q, kappa, layers, seed)}


def run_study(cfg: Config, workers: int = 4) -> dict[str, pd.DataFrame | list]:
    """Run all jobs in parallel and assemble the raw tables."""
    work = [(*j, cfg.maxiter, cfg.p_grid) for j in jobs(cfg)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(run_job, work))
    else:
        outs = [run_job(w) for w in work]
    traj = pd.DataFrame([r for o in outs for r in o["traj"]])
    rows = pd.DataFrame([r for o in outs for r in o["rows"]])
    final = pd.DataFrame([o["final"] for o in outs])
    return {"traj": traj, "analysis": rows, "final": final, "outs": outs}


def shots_table(analysis: pd.DataFrame, final: pd.DataFrame, targets: list[float]) -> pd.DataFrame:
    """Shots to certify each absolute precision target, for both allocations (inf = noise/ansatz floor)."""
    recs = []
    for r in analysis.itertuples(index=False):
        an = Analysis(r.c_mean, r.v_uniform, r.v_neyman, r.n_items, r.gates_per_shot)
        for et in targets:
            tc = (et / r.kappa) ** 2
            for alloc in ("uniform", "neyman"):
                s = shots_to_certify(an, alloc, tc)
                recs.append({"q": r.q, "kappa": r.kappa, "layers": r.layers, "seed": r.seed, "p": r.p,
                             "zne": r.zne, "eps_target": et, "alloc": alloc, "shots": s,
                             "cnot_executions": s * r.gates_per_shot})
    return pd.DataFrame(recs)


def pareto_table(analysis: pd.DataFrame, q: int, kappa: float, layers: int, p: float,
                 shot_ladder: np.ndarray) -> pd.DataFrame:
    """Median 95 % certified precision as a function of the total shot budget, per method."""
    sub = analysis[(analysis.q == q) & (analysis.kappa == kappa) & (analysis.layers == layers) & (analysis.p == p)]
    recs = []
    for r in sub.itertuples(index=False):
        an = Analysis(r.c_mean, r.v_uniform, r.v_neyman, r.n_items, r.gates_per_shot)
        for alloc in ("uniform", "neyman"):
            for s in shot_ladder:
                recs.append({"seed": r.seed, "zne": r.zne, "alloc": alloc, "shots": float(s),
                             "eps_cert": certified_eps(an, float(s), alloc, kappa)})
    return pd.DataFrame(recs)


def mc_validation(outs: list[dict], cfg: Config, eps_t: float = 0.4) -> pd.DataFrame:
    """Binomial shot simulation of the certificate on the base instance, seed 0."""
    rng = np.random.default_rng(2027)
    key = (cfg.base_q, cfg.base_kappa, cfg.base_layers, 0)
    o = next(o for o in outs if o["inst"] == key)
    q, kappa, layers, seed = key
    inst = make_instance(q, kappa, seed)
    ts = build_terms(inst)
    c_ideal = o["final"]["cost"]
    recs = []
    for p in (0.0, 1e-5):
        for zne in (False, True):
            an = analyse(o["mu_h"], o["mu_p"], ts, q, layers, p, zne)
            tc = (eps_t / kappa) ** 2
            s = shots_to_certify(an, "uniform", tc)
            if not np.isfinite(s):
                continue
            per_item = int(np.ceil(s / an.n_items))
            mc = monte_carlo(o["mu_h"], o["mu_p"], ts, q, layers, p, zne, per_item, c_ideal, tc,
                             cfg.mc_reps, rng)
            recs.append({"p": p, "zne": zne, "eps_target": eps_t, "predicted_shots": s,
                         "nominal_issue_rate": 0.95, "nominal_coverage": 0.95, **mc})
    return pd.DataFrame(recs)
