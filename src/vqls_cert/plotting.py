"""Figures (Okabe-Ito palette, 200 dpi): Pareto scatter, annotated heatmaps, empirical CDFs, log-log scaling."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "pink": "#CC79A7"}
DPI = 200


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_pareto(par: pd.DataFrame, eps_true: float, p: float, kappa: float, path: Path) -> None:
    """Cost-vs-quality scatter: total shots against the 95 % certified precision, with the Pareto front."""
    methods = [("uniform", False, "equal shots, raw", OI["blue"], "o"),
               ("neyman", False, "variance-optimal shots, raw", OI["sky"], "s"),
               ("uniform", True, "equal shots, ZNE", OI["verm"], "^"),
               ("neyman", True, "variance-optimal shots, ZNE", OI["orange"], "D")]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    pts = []
    for alloc, zne, lab, col, mk in methods:
        d = par[(par.alloc == alloc) & (par.zne == zne)].groupby("shots").eps_cert.mean().reset_index()
        ax.scatter(d.shots, d.eps_cert, s=34, color=col, marker=mk, label=lab, zorder=3, edgecolor="white", lw=0.4)
        pts += list(zip(d.shots, d.eps_cert))
    pts.sort()
    front, best = [], np.inf
    for s, e in pts:
        if e < best - 1e-12:
            best = e
            front.append((s, e))
    fs, fe = zip(*front)
    ax.step(fs, fe, where="post", color=OI["black"], lw=1.6, label="Pareto front", zorder=2)
    ax.axhline(eps_true, color=OI["green"], ls="--", lw=1.4, label="true error of the trained state")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total shots spent on certification (count)")
    ax.set_ylabel("95 % certified trace-distance bound (dimensionless)")
    ax.set_title(f"Certification cost vs certified precision (4 qubits, kappa = {kappa:g}, p = {p:g})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, which="both")
    _save(fig, path)


def _annot_heat(ax, mat: np.ndarray, xl: list[str], yl: list[str], fmt, norm, cmap="cividis") -> None:
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto", origin="lower")
    ax.set_xticks(range(len(xl)), xl)
    ax.set_yticks(range(len(yl)), yl)
    lo, hi = np.nanmin(mat), np.nanmax(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            frac = (np.log(v) - np.log(lo)) / max(np.log(hi) - np.log(lo), 1e-9) if np.isfinite(v) and v > 0 else 1
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8,
                    color="white" if frac < 0.55 else "black")
    return im


def fig_floor_heatmap(floor: pd.DataFrame, path: Path) -> None:
    """Two annotated heatmaps (raw / ZNE) of the infinite-shot certified-precision floor over p and kappa."""
    ps = sorted(floor.p.unique())
    ks = sorted(floor.kappa.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    norm = LogNorm(vmin=max(floor.eps_floor.min(), 1e-4), vmax=min(floor.eps_floor.max(), 1e3))
    for ax, zne, title in ((axes[0], False, "raw"), (axes[1], True, "with ZNE (lambda = 1, 3, 5)")):
        mat = np.array([[floor[(floor.p == p) & (floor.kappa == k) & (floor.zne == zne)].eps_floor.mean()
                         for k in ks] for p in ps])
        mat = np.clip(mat, norm.vmin, norm.vmax)
        fmt = lambda v: f"{v:.2g}" if v < 100 else ">100"
        im = _annot_heat(ax, mat, [f"{k:g}" for k in ks], [f"{p:g}" for p in ps], fmt, norm)
        ax.set_title(f"Certified-precision floor, {title}")
        ax.set_xlabel("condition number kappa (dimensionless)")
    axes[0].set_ylabel("depolarising error per CNOT, p (probability)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="floor on kappa * sqrt(C) at infinite shots (dimensionless)")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_looseness_ecdf(traj: pd.DataFrame, path: Path) -> None:
    """Empirical CDF of bound / true error along the optimisation, one curve per condition number."""
    cols = [OI["blue"], OI["orange"], OI["green"], OI["verm"], OI["pink"], OI["sky"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for col, (k, d) in zip(cols, traj.groupby("kappa")):
        r = np.sort((d.eps_bound / d.eps_true).values)
        ax.step(r, np.arange(1, len(r) + 1) / len(r), where="post", color=col, lw=1.8, label=f"kappa = {k:g}")
    ax.axvline(1.0, color=OI["black"], ls=":", lw=1.2)
    ax.text(1.02, 0.04, "bound = truth", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("certified bound / true trace distance (dimensionless, log scale)")
    ax.set_ylabel("empirical CDF over optimiser iterates")
    ax.set_title("How loose is the kappa * sqrt(C) certificate? (4 qubits, 3 layers)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    _save(fig, path)


def _geo(d: pd.DataFrame, by: str) -> pd.DataFrame:
    """Geometric mean and +/- 1 standard deviation (in log10) of the finite shot counts per group."""
    rows = []
    for k, g in d[np.isfinite(d.shots)].groupby(by):
        lg = np.log10(g.shots.values)
        sd = lg.std(ddof=1) if len(lg) > 1 else 0.0
        rows.append({by: k, "gm": 10 ** lg.mean(), "lo": 10 ** (lg.mean() - sd), "hi": 10 ** (lg.mean() + sd)})
    return pd.DataFrame(rows, columns=[by, "gm", "lo", "hi"])


def fig_scaling(shots: pd.DataFrame, base_q: int, base_kappa: float, base_layers: int, path: Path) -> None:
    """Log-log scaling of the shots to certify with kappa (left) and with the number of qubits (right)."""
    cols = {0.1: OI["blue"], 0.2: OI["orange"], 0.4: OI["green"], 0.8: OI["verm"]}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    ax = axes[0]
    noise_free = shots[(shots.p == 0.0) & (~shots.zne) & (shots.alloc == "uniform") & (shots.layers == base_layers)]
    d0 = noise_free[noise_free.q == base_q]
    for et, c in cols.items():
        d = d0[d0.eps_target == et]
        g = _geo(d, "kappa")
        if len(g):
            ax.errorbar(g.kappa, g.gm, yerr=[g.gm - g.lo, g.hi - g.gm], color=c, marker="o", capsize=3,
                        label=f"target eps = {et:g}")
    kk = np.array(sorted(d0.kappa.unique()))
    ref = d0[(d0.eps_target == 0.8) & np.isfinite(d0.shots)].groupby("kappa").shots.mean()
    if len(ref):
        ax.plot(kk, ref.iloc[0] * (kk / kk[0]) ** 4, color=OI["black"], ls=":", label="kappa^4 reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("condition number kappa (dimensionless)")
    ax.set_ylabel("total shots to certify, 95 % / 95 % (count)\n(geometric mean, +/- 1 sd of log10)")
    ax.set_title(f"Noise-free certification cost vs kappa ({base_q} qubits)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which="both")
    ax = axes[1]
    d1 = noise_free[noise_free.kappa == base_kappa]
    for et, c in cols.items():
        g = _geo(d1[d1.eps_target == et], "q")
        if len(g):
            ax.errorbar(g.q, g.gm, yerr=[g.gm - g.lo, g.hi - g.gm], color=c, marker="s", capsize=3,
                        label=f"target eps = {et:g}")
    ax.set_yscale("log")
    ax.set_xticks(sorted(d1.q.unique()))
    ax.set_xlabel("number of qubits (count)")
    ax.set_ylabel("total shots to certify, 95 % / 95 % (count)\n(geometric mean, +/- 1 sd of log10)")
    ax.set_title(f"Noise-free certification cost vs size (kappa = {base_kappa:g})")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which="both")
    _save(fig, path)


def fig_depth_heatmap(depth: pd.DataFrame, eps_t: float, path: Path) -> None:
    """Annotated heatmap: log10 shots to certify over ansatz depth and per-CNOT error (infinity = floor)."""
    ps = sorted(depth.p.unique())
    ls = sorted(depth.layers.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    vals = depth[np.isfinite(depth.shots)].shots
    norm = LogNorm(vmin=max(vals.min(), 1.0), vmax=vals.max())
    for ax, zne, title in ((axes[0], False, "raw"), (axes[1], True, "with ZNE")):
        mat = np.full((len(ls), len(ps)), np.nan)
        lab = [[""] * len(ps) for _ in ls]
        for i, L in enumerate(ls):
            for j, p in enumerate(ps):
                d = depth[(depth.layers == L) & (depth.p == p) & (depth.zne == zne)].shots
                fin = d[np.isfinite(d)]
                if len(fin) * 2 > len(d):
                    mat[i, j] = fin.median()
                    lab[i][j] = f"{np.log10(fin.median()):.1f}"
                else:
                    lab[i][j] = "inf"
        shown = np.where(np.isnan(mat), norm.vmax, mat)
        im = ax.imshow(shown, cmap="cividis", norm=norm, aspect="auto", origin="lower")
        ax.set_xticks(range(len(ps)), [f"{p:g}" for p in ps], rotation=45)
        ax.set_yticks(range(len(ls)), [str(L) for L in ls])
        for i in range(len(ls)):
            for j in range(len(ps)):
                dark = np.isfinite(mat[i, j]) and mat[i, j] < np.sqrt(norm.vmin * norm.vmax)
                ax.text(j, i, lab[i][j], ha="center", va="center", fontsize=8, color="white" if dark else "black")
        ax.set_title(f"log10 shots to certify eps = {eps_t:g}, {title}")
        ax.set_xlabel("depolarising error per CNOT, p (probability)")
    axes[0].set_ylabel("ansatz layers (count)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="shots (count, log scale)")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
