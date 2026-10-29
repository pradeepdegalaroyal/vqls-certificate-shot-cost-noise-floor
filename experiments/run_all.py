"""Reproduce every result and figure.  ``python experiments/run_all.py`` (full) or ``--quick`` (smoke run)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqls_cert import plotting as P  # noqa: E402
from vqls_cert.pipeline import (EPS_TARGETS, Config, mc_validation, pareto_table, run_study,  # noqa: E402
                                shots_table)

PARETO_P = 1e-5
HEAT_TARGET = 0.2


def _ms(x: pd.Series) -> dict:
    return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, "n": int(len(x))}


def _ls(x: pd.Series) -> dict:
    """Mean and standard deviation of log10 of the finite values (shot counts span decades)."""
    v = np.log10(np.asarray(x, dtype=float)[np.isfinite(np.asarray(x, dtype=float))])
    return {"log10_mean": float(v.mean()) if len(v) else None,
            "log10_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n_finite": int(len(v)), "n_total": int(len(x))}


def _slope(x: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 3 or len(np.unique(x[ok])) < 2:
        return {"slope": None, "stderr": None, "n": int(ok.sum())}
    r = stats.linregress(np.log10(x[ok]), np.log10(y[ok]))
    return {"slope": float(r.slope), "stderr": float(r.stderr), "n": int(ok.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run, well under one minute")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = Config.quick() if args.quick else Config()
    out_r = ROOT / ("results_quick" if args.quick else "results")
    out_f = ROOT / ("figures_quick" if args.quick else "figures")
    out_r.mkdir(exist_ok=True); out_f.mkdir(exist_ok=True)
    t0 = time.time()

    study = run_study(cfg, workers=args.workers)
    traj, analysis, final = study["traj"], study["analysis"], study["final"]
    shots = shots_table(analysis, final, EPS_TARGETS)
    shots["shots"] = shots["shots"].astype(float)
    traj.to_csv(out_r / "trajectories.csv", index=False)
    analysis.to_csv(out_r / "estimator_moments.csv", index=False)
    final.to_csv(out_r / "trained_states.csv", index=False)
    shots.to_csv(out_r / "shots_to_certify.csv", index=False)
    mc = mc_validation(study["outs"], cfg)
    mc.to_csv(out_r / "monte_carlo_validation.csv", index=False)

    summ: dict = {"config": {"quick": args.quick, "seeds": cfg.seeds, "maxiter": cfg.maxiter,
                             "base_q": cfg.base_q, "base_kappa": cfg.base_kappa, "base_layers": cfg.base_layers}}
    bq, bk, bl = cfg.base_q, cfg.base_kappa, cfg.base_layers

    # -- E1: validity and looseness of the kappa * sqrt(C) bound
    tr = traj[(traj.q == bq) & (traj.layers == bl) & (traj.eps_true > 1e-9)]
    summ["bound_valid_fraction_strict"] = float(np.mean(traj.eps_true <= traj.eps_bound))
    summ["bound_valid_fraction_tol1e-7"] = float(np.mean(traj.eps_true <= traj.eps_bound + 1e-7))
    summ["bound_iterates"] = int(len(traj))
    summ["bound_max_violation"] = float(np.max(traj.eps_true - traj.eps_bound))
    summ["looseness_median_by_kappa"] = {f"{k:g}": float(np.median(d.eps_bound / d.eps_true)) for k, d in tr.groupby("kappa")}
    f_base = final[(final.q == bq) & (final.layers == bl)]
    summ["final_looseness_by_kappa"] = {f"{k:g}": _ms(d.eps_bound / d.eps_true) for k, d in f_base.groupby("kappa")}
    summ["final_cost_by_kappa"] = {f"{k:g}": _ms(d.cost) for k, d in f_base.groupby("kappa")}
    summ["final_eps_true_by_kappa"] = {f"{k:g}": _ms(d.eps_true) for k, d in f_base.groupby("kappa")}

    # -- E2: noise-free shot cost, slopes, allocation ablation
    nf = shots[(shots.p == 0.0) & (~shots.zne)]
    uni = nf[(nf.alloc == "uniform") & (nf.q == bq) & (nf.layers == bl)]
    summ["shots_by_kappa_target"] = {f"{k:g}|{e:g}": _ls(d.shots)
                                     for (k, e), d in uni.groupby(["kappa", "eps_target"])}
    summ["slope_vs_kappa"] = {f"{e:g}": _slope(d.kappa.values, d.shots.values) for e, d in uni.groupby("eps_target")}
    q_sc = nf[(nf.alloc == "uniform") & (nf.kappa == bk) & (nf.layers == bl)]
    summ["shots_by_qubits_target"] = {f"{q}|{e:g}": _ls(d.shots)
                                      for (q, e), d in q_sc.groupby(["q", "eps_target"])}
    ratio = nf.pivot_table(index=["q", "kappa", "layers", "seed", "eps_target"], columns="alloc", values="shots").reset_index()
    ratio = ratio[np.isfinite(ratio.uniform) & (ratio.uniform > 0)]
    ratio["gain"] = ratio.uniform / ratio.neyman
    summ["neyman_gain_overall"] = _ms(ratio.gain)
    summ["neyman_gain_by_q"] = {str(q): _ms(d.gain) for q, d in ratio[ratio.kappa == bk].groupby("q")}

    # -- E3: noise floor and ZNE
    fl = analysis.copy()
    fl["eps_floor"] = fl.kappa * np.sqrt(np.clip(fl.c_mean, 0, None))
    fl_base = fl[(fl.q == bq) & (fl.layers == bl)]
    fl_base.groupby(["p", "kappa", "zne"]).eps_floor.agg(["mean", "std"]).reset_index().to_csv(out_r / "noise_floor.csv", index=False)
    summ["floor_bq_bk"] = {f"{p:g}|{'zne' if z else 'raw'}": _ms(d.eps_floor)
                           for (p, z), d in fl_base[fl_base.kappa == bk].groupby(["p", "zne"])}
    tol = {}
    for z in (False, True):
        okp = [p for p in cfg.p_grid if fl_base[(fl_base.p == p) & (fl_base.zne == z) & (fl_base.kappa == bk)].eps_floor.mean() <= HEAT_TARGET]
        tol["zne" if z else "raw"] = max(okp) if okp else None
    summ["max_p_with_floor_below_target"] = {"target_eps": HEAT_TARGET, **tol}

    # -- depth ablation
    dep = shots[(shots.q == bq) & (shots.kappa == bk) & (shots.alloc == "uniform") & (shots.eps_target == HEAT_TARGET)]
    dep_tab = dep.groupby(["layers", "p", "zne"]).shots.apply(lambda s: float(np.median(s)) if np.isfinite(s).sum() * 2 > len(s) else float("inf")).reset_index()
    dep_tab.to_csv(out_r / "depth_ablation.csv", index=False)
    summ["depth_ablation_noise_free"] = {str(L): {"cost": _ms(d.cost), "eps_true": _ms(d.eps_true), "n_cnots": int(d.n_cnots.iloc[0])}
                                         for L, d in final[(final.q == bq) & (final.kappa == bk)].groupby("layers")}

    # -- classical comparison (noise-free, uniform allocation, eps target = HEAT_TARGET)
    cmp_ = uni[uni.eps_target == HEAT_TARGET].merge(final, on=["q", "kappa", "layers", "seed"])
    cmp_["cert_over_cg"] = cmp_.cnot_executions / cmp_.cg_flops
    summ["classical"] = {f"{k:g}": {"cg_iters": _ms(d.cg_iters), "cg_flops": _ms(d.cg_flops),
                                    "cnot_executions": _ls(d.cnot_executions)}
                         for k, d in cmp_.groupby("kappa")}
    summ["classical_ratio_overall"] = _ls(cmp_.cert_over_cg)
    # cost of mitigation at zero noise and of the noise-induced overhead, same seeds
    pv = shots[(shots.q == bq) & (shots.layers == bl) & (shots.kappa == bk) & (shots.alloc == "uniform") & (shots.eps_target == HEAT_TARGET)]
    pv = pv.pivot_table(index=["seed", "p"], columns="zne", values="shots").reset_index()
    z0 = pv[pv.p == 0.0]
    summ["zne_shot_tax_at_p0"] = _ms(z0[True] / z0[False])
    summ["shots_p0_raw_vs_zne"] = {"raw": _ls(z0[False]), "zne": _ls(z0[True])}
    summ["monte_carlo"] = mc.to_dict(orient="records")

    # -- figures
    ladder = np.logspace(5, 13, 17)
    par = pareto_table(analysis, bq, bk, bl, PARETO_P if PARETO_P in cfg.p_grid else cfg.p_grid[-1], ladder)
    p_used = PARETO_P if PARETO_P in cfg.p_grid else cfg.p_grid[-1]
    P.fig_pareto(par, float(f_base[f_base.kappa == bk].eps_true.mean()), p_used, bk, out_f / "fig1_pareto_shots_vs_certified_precision.png")
    P.fig_floor_heatmap(fl_base, out_f / "fig2_noise_floor_heatmap.png")
    P.fig_looseness_ecdf(tr, out_f / "fig3_bound_looseness_ecdf.png")
    P.fig_scaling(shots, bq, bk, bl, out_f / "fig4_shot_scaling.png")
    P.fig_depth_heatmap(dep, HEAT_TARGET, out_f / "fig5_depth_noise_heatmap.png")
    summ["pareto_p"] = p_used
    summ["runtime_s"] = time.time() - t0
    (out_r / "summary.json").write_text(json.dumps(summ, indent=2, default=lambda o: None if o is None else str(o)), encoding="utf8")
    print(f"done in {time.time() - t0:.0f}s -> {out_r}, {out_f}")


if __name__ == "__main__":
    main()
