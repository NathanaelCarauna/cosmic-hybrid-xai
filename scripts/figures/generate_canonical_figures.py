#!/usr/bin/env python3
"""S4-T3 — Gera as 3 figuras-canhão da dissertação.

fig1: heatmap SA (6 métodos × 4 formatos) com anotações e marcação home/N-A
fig2: scatter pred×GT por formato (E33d/E34a vs R1), diagonais ±25%
fig3: boxplot de erro assinado por faixa CFP × método (De Vito)

Insumos: results/cross-dataset-matrix/summary.csv, error-analysis/per_item_all.csv
Output: figures/fig{1,2,3}-*.{pdf,png}
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 120})

DATASETS = ["devito", "zenodo", "ochodek", "user-stories"]
DS_LABELS = ["De Vito\n(UCs)", "Zenodo\n(sentenças)", "Ochodek\n(nomes UC)", "User Stories\n(Connextra)"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] figures/{name}.{{pdf,png}}")


def fig1_heatmap():
    rows = list(csv.DictReader((BASE / "results/cross-dataset-matrix/summary.csv").open()))
    # summary.csv (wide): method, <ds>_mae, <ds>_sa, <ds>_n, <ds>_note por dataset
    sa = {}
    for r in rows:
        for d in DATASETS:
            v = r.get(f"{d}_sa", "")
            if v not in ("", None):
                try:
                    sa[(r["method"], d)] = float(v)
                except ValueError:
                    pass
    methods = ["hybrid-xai-e33d-adapters", "hybrid-xai-e33d", "r3-hko",
               "r1-cosmet", "hybrid-xai-e30f-clean", "r2-deep-cosmic"]
    labels = ["E33d+E34a\n(adaptadores)", "E33d puro", "R3-HKO\n(within-ds)",
              "R1-COSMET", "E30f-clean", "R2-DEEP-\nCOSMIC"]
    M = np.full((len(methods), 4), np.nan)
    for i, m in enumerate(methods):
        for j, d in enumerate(DATASETS):
            M[i, j] = sa.get((m, d), np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(np.clip(M, -1, 1), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(len(methods)):
        for j in range(4):
            v = M[i, j]
            txt = "N/A" if np.isnan(v) else f"{v:+.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="black" if not np.isnan(v) and abs(v) < 0.75 else
                          ("white" if not np.isnan(v) else "gray"))
    ax.set_xticks(range(4), DS_LABELS)
    ax.set_yticks(range(len(methods)), labels)
    ax.set_title("Standardized Accuracy (SA) por método × formato\n"
                 "(valores < −1 truncados na escala; SA>0 = melhor que predizer a média)")
    fig.colorbar(im, ax=ax, label="SA (truncado em ±1)")
    save(fig, "fig1-heterogeneity-heatmap")


def load_items():
    cells = defaultdict(list)
    with (BASE / "results/error-analysis/per_item_all.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            cells[(r["method"], r["dataset"])].append(
                (float(r["y_true"]), float(r["y_pred"])))
    return cells


def fig2_scatter(cells):
    ours = {"devito": "E33d", "zenodo": "E33d", "ochodek": "E33d",
            "user-stories": "E34a"}
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=False)
    for j, ds in enumerate(DATASETS):
        for row, (label, meth) in enumerate(
                [("Hybrid-XAI (E33d/E34a)", ours[ds]), ("R1-COSMET", "R1-COSMET")]):
            ax = axes[row, j]
            pts = cells.get((meth, ds), [])
            if pts:
                yt = np.array([p[0] for p in pts])
                yp = np.array([p[1] for p in pts])
                jit = (np.random.default_rng(1).normal(0, 0.06, (2, len(yt)))
                       if len(yt) > 300 else np.zeros((2, len(yt))))
                ax.scatter(yt + jit[0], yp + jit[1], s=12, alpha=0.35,
                           color="#2a6fb0" if row == 0 else "#b05a2a",
                           edgecolors="none")
                lim = max(yt.max(), yp.max()) * 1.05 + 0.5
                x = np.linspace(0, lim, 50)
                ax.plot(x, x, "k-", lw=0.8)
                ax.fill_between(x, x * 0.75, x * 1.25, color="gray", alpha=0.15)
                ax.set_xlim(0, lim); ax.set_ylim(0, lim)
                mae = np.abs(yt - yp).mean()
                ax.set_title(f"{DS_LABELS[j].replace(chr(10), ' ')} — {label}\n"
                             f"n={len(yt)}, MAE={mae:.2f}", fontsize=9)
            ax.set_xlabel("CFP real (GT)")
            if j == 0:
                ax.set_ylabel("CFP predito")
    fig.suptitle("Predição × Ground Truth por formato (faixa cinza = ±25%)", y=1.0)
    fig.tight_layout()
    save(fig, "fig2-pred-vs-gt-scatter")


def fig3_residual_boxplot(cells):
    buckets = [("1-2", lambda c: c <= 2), ("3-4", lambda c: 3 <= c <= 4),
               ("5-9", lambda c: 5 <= c <= 9), ("10+", lambda c: c >= 10)]
    methods = [("E33d", "#2a6fb0"), ("R1-COSMET", "#b05a2a"), ("E30f", "#3f9b56")]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    for mi, (meth, color) in enumerate(methods):
        pts = cells.get((meth, "devito"), [])
        data, positions = [], []
        for bi, (bl, cond) in enumerate(buckets):
            vals = [p[1] - p[0] for p in pts if cond(p[0])]
            if vals:
                data.append(vals)
                positions.append(bi + (mi - 1) * width)
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(markersize=3, alpha=0.5))
        for box in bp["boxes"]:
            box.set_facecolor(color); box.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("black")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(4), [b[0] for b in buckets])
    ax.set_xlabel("Faixa de CFP real (GT)")
    ax.set_ylabel("Erro assinado (pred − GT)")
    ax.set_title("Erro por faixa de complexidade — De Vito\n"
                 "(todos os métodos subcontam em GT≥10 — modo de falha M1)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.6) for _, c in methods]
    ax.legend(handles, [m for m, _ in methods], loc="lower left")
    fig.tight_layout()
    save(fig, "fig3-residual-by-bucket")


def main():
    print("Gerando figuras canônicas...")
    fig1_heatmap()
    cells = load_items()
    fig2_scatter(cells)
    fig3_residual_boxplot(cells)


if __name__ == "__main__":
    main()
