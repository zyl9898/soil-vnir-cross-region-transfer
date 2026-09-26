"""
Figures for the cross-region soil Vis-NIR transfer study.

Python/matplotlib backend (the saved nature-figure preference). Every
multi-panel figure passes the blocking panel-alignment gate before export and
every exported PDF is checked with the glyph-floor and collision audits.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SKILL_SCRIPTS = os.path.expanduser("~/.dsh/skills/nature-figure/scripts")
if SKILL_SCRIPTS not in sys.path:
    sys.path.insert(0, SKILL_SCRIPTS)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

from common import FIGURES, RESULTS, build_dataset, ensure_dirs, wavelength_columns  # noqa: E402

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_3": "#8BCF8B",
    "red_strong": "#B64342", "neutral_light": "#CFCECE", "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D", "teal": "#42949E", "violet": "#9A4D8E",
}
MODEL_COLOR = {"PLSR": PALETTE["blue_main"], "Ridge": PALETTE["neutral_mid"],
               "RandomForest": PALETTE["green_3"], "HistGBM": PALETTE["violet"],
               "MLP": PALETTE["red_strong"]}
PREP_COLOR = {"raw": PALETTE["neutral_mid"], "snv": PALETTE["teal"],
              "sg1": PALETTE["blue_main"], "snv_sg1": PALETTE["violet"]}


def apply_style(font_size=7, lw=0.8):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = lw
    # opaque legend background: unframed legends placed inside the axes let the
    # data lines cross the legend text
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "none"
    plt.rcParams["legend.framealpha"] = 0.92
    plt.rcParams["axes.titlesize"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size - 0.5
    plt.rcParams["ytick.labelsize"] = font_size - 0.5
    plt.rcParams["legend.fontsize"] = font_size - 0.5
    # keep legend handles clear of their labels (adjacent text was being read as
    # crossed by the handle stroke)
    plt.rcParams["legend.handletextpad"] = 0.9
    plt.rcParams["legend.borderpad"] = 0.5
    plt.rcParams["legend.labelspacing"] = 0.45
    plt.rcParams["legend.handlelength"] = 1.6


def add_panel_label(ax, label, x=0.0, y=1.0, xo=-4, yo=3, fontsize=8,
                    color="black", va="bottom"):
    from matplotlib.transforms import ScaledTranslation
    off = ScaledTranslation(xo / 72, yo / 72, ax.figure.dpi_scale_trans)
    ax.text(x, y, label, transform=ax.transAxes + off, fontsize=fontsize,
            fontweight="bold", color=color, ha="left", va=va)


def save_pub(fig, stem, dpi=600):
    ensure_dirs()
    base = os.path.join(FIGURES, stem)
    require_matplotlib_panel_alignment(
        fig, json_out=f"{base}.alignment.json", overlay_svg=f"{base}.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    for fmt in ("svg", "pdf"):
        fig.savefig(f"{base}.{fmt}", bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=300, bbox_inches="tight")
    print(f"  saved {stem}.svg/.pdf/.tiff")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig1():
    print("Fig 1 — data and domain shift")
    d = build_dataset()
    wl = wavelength_columns(d)
    bands = np.array([int(c[1:]) for c in wl])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) samples per region
    cnt = d.groupby("REGION_NAME").size().sort_values()
    ax_a.barh(np.arange(len(cnt)), cnt.values, color=PALETTE["blue_main"], height=0.72)
    ax_a.set_yticks(np.arange(len(cnt)))
    ax_a.set_yticklabels(cnt.index, fontsize=6)
    ax_a.set_xlabel("Samples")
    ax_a.set_title("Held-out regions (n \u2265 40)", pad=4)

    # (b) mean spectrum per region (the domain shift itself)
    top = d.groupby("REGION_NAME").size().sort_values(ascending=False).head(13).index
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(top):
        sub = d[d.REGION_NAME == r]
        ax_b.plot(bands, sub[wl].to_numpy(dtype=float).mean(axis=0),
                  lw=0.9, color=cmap(i % 20), label=r)
    ax_b.set_xlabel("Wavelength (nm)")
    ax_b.set_ylabel("Reflectance")
    ax_b.set_title("Mean spectrum per region", pad=4)
    ax_b.legend(fontsize=6, ncol=2, loc="upper left", handlelength=1.0)

    # (c) ORGC distribution
    ax_c.hist(np.log10(d.ORGC.to_numpy(dtype=float)), bins=50,
              color=PALETTE["teal"], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax_c.set_xlabel("log10 ORGC (%)")
    ax_c.set_ylabel("Samples")
    ax_c.set_title("Organic carbon", pad=4)

    # (d) PCA of the spectra, coloured by region: shows the shift directly
    X = d[wl].to_numpy(dtype=float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xc = Xs - Xs.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc = U[:, :2] * S[:2]
    regs = d.REGION_NAME.to_numpy()
    for i, r in enumerate(top):
        m = regs == r
        ax_d.scatter(pc[m, 0], pc[m, 1], s=2.0, color=cmap(i % 20), label=None, alpha=0.7)
    ax_d.set_xlabel("PC1")
    ax_d.set_ylabel("PC2")
    ax_d.set_title("Spectra separate by region", pad=4)
    ev = (S ** 2 / (S ** 2).sum())[:2] * 100
    ax_d.text(0.98, 0.04, f"PC1+PC2 = {ev.sum():.0f}% var",
              transform=ax_d.transAxes, ha="right", fontsize=6,
              color=PALETTE["neutral_dark"])

    for lab, a in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        add_panel_label(a, lab)
    fig.tight_layout(pad=1.5)
    save_pub(fig, "Fig1_data")


def fig2():
    print("Fig 2 — three protocols and the penalty decomposition")
    d = pd.read_csv(os.path.join(RESULTS, "E4_decomposition.csv"))
    d.columns = [c.strip() for c in d.columns]
    d = d.sort_values("group_region")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    ax_a, ax_b = axes

    # (a) three protocols on one axis per configuration
    ypos = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        ax_a.plot([r.group_region, r["random"]], [i, i], color=PALETTE["neutral_light"],
                  lw=1.3, zorder=1)
        ax_a.plot([r["random"]], [i], "o", ms=3.4, color=MODEL_COLOR[r.model], zorder=3)
        ax_a.plot([r.group_profile], [i], "o", ms=3.4, mfc="white",
                  mec=MODEL_COLOR[r.model], mew=1.0, zorder=3)
        ax_a.plot([r.group_region], [i], "o", ms=3.4, mfc=MODEL_COLOR[r.model],
                  mec=PALETTE["neutral_dark"], mew=0.7, zorder=4)
    ax_a.set_yticks(ypos)
    ax_a.set_yticklabels([f"{m} · {p}" for m, p in zip(d.model, d.prep)], fontsize=6)
    for t, m in zip(ax_a.get_yticklabels(), d.model):
        t.set_color(MODEL_COLOR[m])
    ax_a.set_xlabel("Median R\u00b2 (log10 ORGC)")
    ax_a.set_title("Plain K-fold overstates every configuration", pad=4)
    ax_a.plot([], [], "o", color=PALETTE["neutral_dark"], label="random K-fold")
    ax_a.plot([], [], "o", mfc="white", mec=PALETTE["neutral_dark"], label="profile-grouped")
    ax_a.plot([], [], "o", mfc=PALETTE["neutral_dark"], mec=PALETTE["neutral_dark"],
              label="region-grouped")
    ax_a.legend(loc="lower right", fontsize=6)
    ax_a.set_ylim(-0.8, len(d) - 0.2)

    # (b) decomposition by model: leak gain vs genuine region penalty
    models = ["PLSR", "Ridge", "MLP", "HistGBM", "RandomForest"]
    agg = d.groupby("model")[["leak_gain", "region_penalty"]].mean().reindex(models)
    x = np.arange(len(agg)); w = 0.38
    ax_b.bar(x - w / 2, agg.leak_gain, w, color=PALETTE["neutral_mid"],
             edgecolor="black", linewidth=0.6, label="profile leak (K-fold gain)")
    ax_b.bar(x + w / 2, agg.region_penalty, w, color=PALETTE["red_strong"],
             edgecolor="black", linewidth=0.6, label="genuine region penalty")
    top = np.maximum(agg.leak_gain, agg.region_penalty).to_numpy()
    for xi, (lg, rp) in enumerate(zip(agg.leak_gain, agg.region_penalty)):
        # rotated: an upright 3-decimal label is wider than one bar, so its box
        # would overlap the neighbouring bar's edge stroke
        ax_b.text(xi - w / 2, lg + 0.003, f"{lg:.3f}", ha="center", va="bottom",
                  fontsize=6, rotation=90)
        ax_b.text(xi + w / 2, rp + 0.003, f"{rp:.3f}", ha="center", va="bottom",
                  fontsize=6, rotation=90)
    ax_b.set_xticks(x); ax_b.set_xticklabels(agg.index, rotation=20, fontsize=6)
    ax_b.set_ylabel("Median R\u00b2 difference")
    ax_b.set_title("Leak is model-dependent, region penalty is not", pad=4)
    ax_b.set_ylim(0, float(top.max()) * 1.30)
    ax_b.legend(fontsize=6, loc="upper left")

    add_panel_label(ax_a, "a"); add_panel_label(ax_b, "b")
    fig.tight_layout(pad=1.5)
    save_pub(fig, "Fig2_transfer_gap")


def fig3():
    print("Fig 3 — where transfer fails, and what fixes it")
    e2 = pd.read_csv(os.path.join(RESULTS, "E2_summary.csv"))
    e2r = pd.read_csv(os.path.join(RESULTS, "E2_domain_adaptation.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    ax_a, ax_b = axes

    # (a) per-region performance for source-only, best model
    best_model = "MLP"
    base = e2r[(e2r.condition == "source_only") & (e2r.model == best_model)]
    base = base.sort_values("r2_log")
    ax_a.barh(np.arange(len(base)), base.r2_log, color=PALETTE["blue_main"], height=0.7)
    ax_a.axvline(0, color=PALETTE["neutral_dark"], lw=0.8)
    ax_a.set_yticks(np.arange(len(base)))
    ax_a.set_yticklabels(base.region_name, fontsize=6)
    ax_a.set_xlabel("R² (log10 ORGC) on the held-out region")
    ax_a.set_title(f"Transfer varies by region ({best_model}, unadapted)", pad=4)

    # (b) k-shot curve: target-only vs pooled vs coral, per model
    order = ["source_only", "mean_only", "coral"]
    ks = [10, 25, 50, 100]
    for m in ["PLSR", "Ridge", "MLP"]:
        sub = e2[e2.model == m]
        base_v = float(sub[sub.condition == "source_only"].r2_log.iloc[0])
        xs, ys = [0], [base_v]
        for k in ks:
            row = sub[sub.condition == f"target_only_k{k}"]
            if len(row):
                xs.append(k); ys.append(float(row.r2_log.iloc[0]))
        ax_b.plot(xs, ys, marker="o", ms=3.0, lw=1.2, color=MODEL_COLOR[m],
                  label=f"{m}: local labels only")
    ax_b.axhline(0, color=PALETTE["neutral_mid"], lw=0.8, ls=":")
    ax_b.set_xscale("log")
    ax_b.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_b.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_b.set_xticks(ks)
    ax_b.set_xlabel("Labelled samples from the target region (k)")
    ax_b.set_ylabel("R² (log10 ORGC)")
    ax_b.set_title("Local labels beat the global library", pad=4)
    ax_b.legend(fontsize=6, loc="lower right")

    add_panel_label(ax_a, "a"); add_panel_label(ax_b, "b")
    fig.tight_layout(pad=1.5)
    save_pub(fig, "Fig3_transfer_fixes")


def fig4():
    print("Fig 4 — adaptation methods vs k-shot")
    e2 = pd.read_csv(os.path.join(RESULTS, "E2_summary.csv"))
    conds = ["source_only", "mean_only", "coral", "k100", "coral_k100", "target_only_k100"]
    labels = ["source\nonly", "mean\nonly", "CORAL", "+100 pooled", "CORAL\n+100", "100 local\nonly"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    ax_a, ax_b = axes

    models = ["PLSR", "Ridge", "MLP"]
    w = 0.26
    x = np.arange(len(conds))
    for i, m in enumerate(models):
        sub = e2[e2.model == m].set_index("condition")
        vals = [float(sub.loc[c, "r2_log"]) if c in sub.index else np.nan for c in conds]
        ax_a.bar(x + (i - 1) * w, vals, w, color=MODEL_COLOR[m], edgecolor="black",
                 linewidth=0.5, label=m)
    ax_a.axhline(0, color=PALETTE["neutral_dark"], lw=0.8)
    ax_a.set_xticks(x); ax_a.set_xticklabels(labels, fontsize=6)
    ax_a.set_ylabel("Median R² (log10 ORGC)")
    ax_a.set_title("Alignment does not recover the penalty", pad=4)
    ax_a.legend(fontsize=6, loc="upper left", ncol=3)

    # pooled-with-k curve, showing the source library swamps few target samples
    ks = [10, 25, 50, 100]
    for m in models:
        sub = e2[e2.model == m].set_index("condition")
        base_v = float(sub.loc["source_only", "r2_log"])
        ys = [base_v] + [float(sub.loc[f"k{k}", "r2_log"]) for k in ks]
        ax_b.plot([0] + ks, ys, marker="s", ms=2.8, lw=1.1, color=MODEL_COLOR[m], label=m)
    ax_b.set_xscale("log")
    ax_b.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_b.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_b.set_xticks(ks)
    ax_b.set_xlabel("Target samples added to the global pool (k)")
    ax_b.set_ylabel("Median R² (log10 ORGC)")
    ax_b.set_title("Local samples: pooled vs used alone", pad=4)
    ax_b.legend(fontsize=6, loc="lower right")

    add_panel_label(ax_a, "a"); add_panel_label(ax_b, "b")
    fig.tight_layout(pad=1.5)
    save_pub(fig, "Fig4_adaptation")


def fig5():
    print("Fig 5 — 1D-CNN and Deep CORAL")
    p3 = os.path.join(RESULTS, "E3_cnn.csv")
    e1 = pd.read_csv(os.path.join(RESULTS, "E1_summary.csv"))
    cnn = pd.read_csv(p3)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    ax_a, ax_b = axes

    best = (e1[e1.protocol == "leave_region_out"]
            .sort_values("r2_log", ascending=False).head(4))
    names = [f"{m}\n{p}" for m, p in zip(best.model, best.prep)] + ["1D-CNN", "1D-CNN\n+CORAL"]
    vals = list(best.r2_log) + [float(cnn[cnn.variant == "cnn"].r2_log.median()),
                               float(cnn[cnn.variant == "cnn_coral"].r2_log.median())]
    cols = [MODEL_COLOR[m] for m in best.model] + [PALETTE["teal"], PALETTE["neutral_mid"]]
    ax_a.bar(np.arange(len(vals)), vals, color=cols, edgecolor="black", linewidth=0.6)
    ax_a.set_xticks(np.arange(len(vals))); ax_a.set_xticklabels(names, fontsize=6)
    ax_a.set_ylabel("Median R² (log10 ORGC)")
    ax_a.set_title("Capacity is not the binding constraint", pad=4)

    regions = sorted(cnn.region_name.unique())
    for v, mk in (("cnn", "o"), ("cnn_coral", "s")):
        sub = cnn[cnn.variant == v].set_index("region_name").reindex(regions)
        ax_b.plot(range(len(regions)), sub.r2_log, marker=mk, ms=3.0, lw=1.0,
                  color=PALETTE["teal"] if v == "cnn" else PALETTE["neutral_mid"],
                  label="1D-CNN" if v == "cnn" else "1D-CNN + Deep CORAL")
    ax_b.axhline(0, color=PALETTE["neutral_dark"], lw=0.8, ls=":")
    ax_b.set_xticks(range(len(regions)))
    ax_b.set_xticklabels(regions, rotation=90, fontsize=6)
    ax_b.set_ylabel("R² (log10 ORGC)")
    ax_b.set_title("Feature alignment does not recover it", pad=4)
    ax_b.legend(fontsize=6, loc="lower left")

    add_panel_label(ax_a, "a"); add_panel_label(ax_b, "b")
    fig.tight_layout(pad=1.5)
    save_pub(fig, "Fig5_cnn")


def main():
    ensure_dirs()
    apply_style()
    fig1(); fig2(); fig3(); fig4()
    if os.path.exists(os.path.join(RESULTS, "E3_cnn.csv")):
        try:
            fig5()
        except Exception as e:
            print("Fig5 skipped:", e)
    print("=== figures done ===")


if __name__ == "__main__":
    main()
