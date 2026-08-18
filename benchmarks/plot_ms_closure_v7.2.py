#!/usr/bin/env python3
"""Main-text + SI figures for Methodological Closure / **MS + SI v7.2**.

Paired with ``MS_Methods_Results_zh_v7.2.md`` and live ``SI/`` (freeze ``SI/versions/v7.2/``).
Legacy four-model SI (v7): ``benchmarks/generate_si_v7.py`` / ``plot_ms_results_v7.py``.

Original docstring:
Main-text + SI figures for Methodological Closure v1 / MS v7.

Data: outputs/methodological_closure_full_v1/metrics.csv
      (fallback: results/methodological_closure_full_v1/ for the repro package)
      curves: outputs/phase5_curves_20260715/ or results/curves/
Output: figures/ms_results/v7/ (Fig1–5 main-text; Fig4 collage locked;
        Fig4_A/B/C* panels regenerated for manual restitching)
        SI/figures/ (FigS1–S5 renumbered to citation order; S1 e2e, S5 multi-metrics plane)
        SI/tables/ (canonical TableS1–S14; legacy freeze-λ under _archive_legacy/)
        Does not overwrite polished SI/SI.md prose.

Usage:
  python benchmarks/plot_ms_closure_v7.2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]


def _first_existing(*candidates: Path) -> Path:
    """Prefer lab ``outputs/``; fall back to published ``results/`` (repro package)."""
    for p in candidates:
        if p.is_dir() or p.is_file():
            return p
    return candidates[0]


# Lab writes under outputs/; the GitHub repro ships frozen tables under results/.
CLOSURE = _first_existing(
    ROOT / "outputs" / "methodological_closure_full_v1",
    ROOT / "results" / "methodological_closure_full_v1",
    ROOT / "results" / "published" / "methodological_closure_full_v1",
)
OUT_MS = ROOT / "figures" / "ms_results" / "v7"
OUT_SI = ROOT / "SI" / "figures"
# Canonical SI tables live flat under SI/tables/ (no tables/closure/ dual tree).
OUT_TAB = ROOT / "SI" / "tables"
CURVE_DIR = _first_existing(
    ROOT / "outputs" / "phase5_curves_20260715",
    ROOT / "results" / "curves",
    ROOT / "results" / "published" / "phase5_curves_20260715",
)

REGION_ORDER = ["AWT", "CAN", "NSW", "NZ", "SA", "SWI"]

# Colour system from docs/colour.png (swatch RGB → hex)
# pink #F4AAD6, sky #94CDDE, purple #967FC5, yellow #F9E498, mint #9BD2CD,
# gray #D7D7D7, lilac #CCBFE3, coral #EF7A64, slate #566C7E, teal #46B0A7
PALETTE = {
    "pink": "#F4AAD6",
    "sky": "#94CDDE",
    "purple": "#967FC5",
    "yellow": "#F9E498",
    "mint": "#9BD2CD",
    "gray": "#D7D7D7",
    "lilac": "#CCBFE3",
    "coral": "#EF7A64",
    "slate": "#566C7E",
    "teal": "#46B0A7",
}

# Six regions: prefer higher-chroma swatches; yellow kept for SA (light → use with edges)
REGION_COLOR = {
    "AWT": PALETTE["sky"],
    "CAN": PALETTE["coral"],
    "NSW": PALETTE["teal"],
    "NZ": PALETTE["purple"],
    "SA": PALETTE["pink"],
    "SWI": PALETTE["slate"],
}

# Method colours: darker hues orthogonal to regional pastels (do not reuse sky/coral/teal/purple/pink/slate).
COLOR_ADDITIVE = "#1B4F72"  # deep navy — B-spline IPP / additive
COLOR_GAM = "#5B2C6F"  # deep plum — same-basis GAM
COLOR_MAXNET = "#922B21"  # brick — maxnet
COLOR_DEEP2_RPHI = "#1B4F72"  # Rφ: navy, solid
COLOR_DEEP2_RX = "#922B21"  # Rx: brick, hatched
COLOR_DEEP3_RPHI = "#1B4F72"
COLOR_DEEP3_RX = "#922B21"
COLOR_RANK_IPP = COLOR_ADDITIVE
COLOR_RANK_MAX = COLOR_MAXNET
COLOR_RANK_FILL_IPP = COLOR_ADDITIVE
COLOR_RANK_FILL_MAX = COLOR_MAXNET


def _setup_style() -> None:
    # Prefer true Times New Roman (Windows TTF on this host); fall back to Liberation/Nimbus.
    tnr = ["Times New Roman", "TimesNewRoman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": tnr,
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 1.25,
            "axes.titleweight": "bold",
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Pin the first available face so matplotlib does not silently swap to DejaVu.
    from matplotlib import font_manager as _fm

    chosen = None
    for name in tnr:
        try:
            path = _fm.findfont(name, fallback_to_default=False)
        except Exception:
            path = ""
        if path and "dejavu" not in path.lower():
            chosen = name
            break
    if chosen is None:
        chosen = "Times New Roman"
    mpl.rcParams["font.family"] = chosen
    mpl.rcParams["font.serif"] = [chosen] + [n for n in tnr if n != chosen]


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            out_dir / f"{name}.{ext}",
            format=ext,
            dpi=600 if ext == "pdf" else 300,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            edgecolor="none",
        )
    plt.close(fig)
    print("wrote", out_dir / name)


def _despine(ax, spine_lw: float = 1.35, tick_lw: float = 1.2, tick_len: float = 4.5) -> None:
    """Hide top/right spines; thicken remaining axes and major ticks for print clarity."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_linewidth(spine_lw)
        ax.spines[side].set_color("0.12")
    ax.tick_params(axis="both", which="major", width=tick_lw, length=tick_len)


def _srgb_luminance(color) -> float:
    """Relative luminance in [0, 1] (W3C sRGB)."""
    r, g, b = mcolors.to_rgb(color)
    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _text_on_face(face, *, thresh: float = 0.22) -> str:
    """White glyphs on dark method fills (navy/plum/brick/slate); dark on pastels."""
    if face is None:
        return "0.15"
    return "white" if _srgb_luminance(face) < thresh else "0.15"


def _boxplot_stats(y) -> dict | None:
    """Median / hinges / Tukey whiskers (same fence as matplotlib boxplot)."""
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    q1, med, q3 = np.percentile(arr, [25.0, 50.0, 75.0])
    iqr = q3 - q1
    lo_lim, hi_lim = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside_lo = arr[arr >= lo_lim]
    inside_hi = arr[arr <= hi_lim]
    return {
        "med": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "whisk_lo": float(inside_lo.min()) if inside_lo.size else float(arr.min()),
        "whisk_hi": float(inside_hi.max()) if inside_hi.size else float(arr.max()),
    }


def _annotate_boxplot_medians(
    ax,
    data_list,
    *,
    positions=None,
    fmt: str = "{:.3f}",
    fontsize: float = 7.0,
    color: str | None = None,
    dy_frac: float = 0.02,
    inside: bool = False,
    face_colors=None,
    placement: str = "auto",
    pad_ylim: bool = True,
) -> None:
    """Label each box with its median (3 d.p. by default).

    placement
        ``"auto"``     — white (or dark) text at the median when the box is tall
                         and the median is away from y = 0; otherwise sit just
                         above the upper whisker so labels leave the zero line
                         and do not sit on a dark fill.
        ``"median"``   — always at the median (contrast from ``face_colors``).
        ``"outside"``  — always above the upper whisker (dark text + white halo).
        ``"stagger_up"`` — both labels above the whisker, even/odd rows offset
                         so adjacent Rφ/Rx pairs do not collide (Fig. 5 A/B).
    """
    stats = [_boxplot_stats(y) for y in data_list]
    if positions is None:
        positions = list(range(1, len(stats) + 1))
    if face_colors is None:
        face_colors = [None] * len(stats)
    else:
        face_colors = list(face_colors)
        if len(face_colors) < len(stats):
            face_colors = face_colors + [None] * (len(stats) - len(face_colors))

    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    # ~one line of 7–8 pt type in axis units, plus a little air.
    line_h = max(0.032 * span, 0.006)
    near_zero = max(0.014, 0.07 * span)
    short_box = 0.12 * span
    outside_pad = max(0.70 * line_h, 0.018)

    planned: list[tuple | None] = []
    for i, (st, x, fc) in enumerate(zip(stats, positions, face_colors)):
        if st is None or not np.isfinite(st["med"]):
            planned.append(None)
            continue
        m, q1, q3 = st["med"], st["q1"], st["q3"]
        mode = placement
        if mode == "auto":
            too_short = (q3 - q1) < short_box
            on_zero = abs(m) < near_zero
            mode = "outside" if (too_short or on_zero) and not inside else "median"
        if mode == "stagger_up":
            extra = (0.0 if i % 2 == 0 else 1.35 * line_h)
            y = st["whisk_hi"] + outside_pad + extra
            # Tiny boxes sitting on y = 0: lift the pair off the dashed line.
            if abs(st["whisk_hi"]) < 0.02 and y < 0.028:
                y = 0.028 + extra
            va = "bottom"
            tc = color or "0.15"
            halo = True
        elif mode == "outside":
            y = st["whisk_hi"] + outside_pad
            if abs(y) < 0.022:
                y = 0.022 if y >= 0 else -0.022
            va = "bottom"
            # If that would leave the top, try below the lower whisker.
            if y > ymax - 0.02 * span and (st["whisk_lo"] - outside_pad) > ymin + 0.02 * span:
                y = st["whisk_lo"] - outside_pad
                va = "top"
            tc = color or "0.15"
            halo = True
        else:
            # Inside / at-median: sit just above the median stroke so glyphs
            # are not bisected; flip below if the upper hinge is tight.
            if inside:
                y, va = m, "center"
            else:
                y = m + dy_frac * span
                va = "bottom"
                if y > q3 - 0.15 * max(q3 - q1, 1e-6) or y > ymax - 0.02 * span:
                    y = m - dy_frac * span
                    va = "top"
            tc = color or _text_on_face(fc)
            halo = False
        planned.append((x, y, va, tc, m, halo))

    if pad_ylim:
        ys = [p[1] for p in planned if p is not None]
        if ys:
            lo, hi = min(ys), max(ys)
            new_lo, new_hi = ymin, ymax
            if lo < ymin + 0.03 * span:
                new_lo = lo - 0.70 * line_h
            if hi > ymax - 0.04 * span:
                new_hi = hi + 1.15 * line_h
            if new_lo != ymin or new_hi != ymax:
                ax.set_ylim(new_lo, new_hi)

    for item in planned:
        if item is None:
            continue
        x, y, va, tc, m, halo = item
        kw = dict(
            ha="center",
            va=va,
            fontsize=fontsize,
            color=tc,
            clip_on=False,
            zorder=6,
        )
        if halo and tc not in ("white", "1.0", "#ffffff", "#FFFFFF"):
            # Dark type only: white halo clears whiskers / the dashed zero line.
            # White-on-fill labels stay unstroked.
            kw["path_effects"] = [pe.withStroke(linewidth=2.6, foreground="white")]
        ax.text(x, y, fmt.format(m), **kw)


def _annotate_region_pair_medians(
    ax,
    data_list,
    *,
    tick_pos,
    face_colors,
    fmt: str = "{:+.3f}",
    fontsize: float = 7.5,
) -> None:
    """Two-line median labels centred on each region (Rφ then Rx).

    Used for Fig. 5 A/B: paired boxes are too close for side-by-side numbers,
    and medians sit on y = 0, so in-box labels look like a misaligned zero line.
    """
    if len(data_list) != 2 * len(tick_pos):
        raise ValueError("expected two boxes (Rφ, Rx) per region tick")
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    line_h = max(0.034 * span, 0.007)
    pad = max(0.55 * line_h, 0.010)

    items = []
    y_hi = ymax
    for k, x_reg in enumerate(tick_pos):
        st0 = _boxplot_stats(data_list[2 * k])
        st1 = _boxplot_stats(data_list[2 * k + 1])
        if st0 is None or st1 is None:
            continue
        anchor = max(st0["whisk_hi"], st1["whisk_hi"]) + pad
        # Keep the stack off the dashed zero line when both boxes are tiny.
        if abs(max(st0["whisk_hi"], st1["whisk_hi"])) < 0.018:
            anchor = max(anchor, 0.026)
        items.append(
            (
                x_reg,
                anchor,
                st0["med"],
                st1["med"],
                face_colors[2 * k],
                face_colors[2 * k + 1],
            )
        )
        y_hi = max(y_hi, anchor + 2.15 * line_h)
    if y_hi > ymax - 0.02 * span:
        ax.set_ylim(ymin, y_hi)

    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    for x, y0, m0, m1, c0, c1 in items:
        ax.text(
            x, y0 + 1.05 * line_h, fmt.format(m0),
            ha="center", va="bottom", fontsize=fontsize,
            color=c0, clip_on=False, zorder=6, fontweight="medium",
            path_effects=halo,
        )
        ax.text(
            x, y0, fmt.format(m1),
            ha="center", va="bottom", fontsize=fontsize,
            color=c1, clip_on=False, zorder=6, fontweight="medium",
            path_effects=halo,
        )


def _annotate_bar_values(
    ax,
    values,
    *,
    positions=None,
    orient: str = "v",
    fmt: str = "{:.3f}",
    fontsize: float = 7.0,
    color: str = "0.15",
    offset_frac: float = 0.015,
) -> None:
    """Label bar heights (vertical) or widths (horizontal)."""
    vals = np.asarray(values, dtype=float)
    if positions is None:
        positions = np.arange(len(vals))
    if orient == "v":
        ymin, ymax = ax.get_ylim()
        span = max(abs(ymax - ymin), 1e-6)
        for x, v in zip(positions, vals):
            if not np.isfinite(v):
                continue
            off = offset_frac * span * (1 if v >= 0 else -1)
            ax.text(
                x,
                v + off,
                fmt.format(v),
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=fontsize,
                color=color,
                clip_on=False,
                zorder=6,
            )
    else:
        xmin, xmax = ax.get_xlim()
        span = max(abs(xmax - xmin), 1e-6)
        for y, v in zip(positions, vals):
            if not np.isfinite(v):
                continue
            off = offset_frac * span * (1 if v >= 0 else -1)
            ax.text(
                v + off,
                y,
                fmt.format(v),
                ha="left" if v >= 0 else "right",
                va="center",
                fontsize=fontsize,
                color=color,
                clip_on=False,
                zorder=6,
            )


def _boot_mean_ci(x, B: int = 2000, seed: int = 0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.array(
        [rng.choice(x, size=len(x), replace=True).mean() for _ in range(B)]
    )
    return float(x.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def load_metrics() -> pd.DataFrame:
    path = CLOSURE / "metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def stage_a_wide(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df.stage == "A"].copy()
    # keep evaluable species with finite additive
    add = a[a.model == "additive_kan_ipp"]
    ok = add[np.isfinite(add.auc_roc)][["region", "species_id", "n_presence", "taxon_group", "lambda_s", "lambda_selection"]]
    wide = a.pivot_table(
        index=["region", "species_id"], columns="model", values="auc_roc", aggfunc="first"
    )
    wide = wide.reset_index().merge(ok, on=["region", "species_id"], how="inner")
    return wide


def e2e_itt(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.stage == "D"].copy()
    picks = []
    order = {"primary": 0, "R1": 1, "R2": 2}
    for (reg, sp), g in d.groupby(["region", "species_id"]):
        ok = g[g["failed"] == False] if "failed" in g.columns else g
        if len(ok):
            ok = ok.copy()
            ok["_o"] = ok["remediation"].map(lambda x: order.get(str(x), 9))
            picks.append(ok.sort_values("_o").iloc[0])
        else:
            picks.append(g.iloc[-1])
    return pd.DataFrame(picks)


def deep_species_mean(df: pd.DataFrame, stage: str, model: str) -> pd.Series:
    sub = df[(df.stage == stage) & (df.model == model)]
    return sub.groupby(["region", "species_id"])["auc_roc"].mean()


def plot_fig1_e2e_maxnet(df: pd.DataFrame) -> None:
    e2e = e2e_itt(df)
    mx = (
        df[(df.stage == "A") & (df.model == "maxnet_bg10k")]
        .set_index(["region", "species_id"])["auc_roc"]
    )
    e2e = e2e.set_index(["region", "species_id"])
    common = e2e.index.intersection(mx.index)
    x = mx.loc[common].values
    y = e2e.loc[common, "auc_roc"].values
    regs = [i[0] for i in common]
    failed = e2e.loc[common, "failed"].values if "failed" in e2e.columns else np.zeros(len(common))

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for reg in REGION_ORDER:
        m = np.array([r == reg for r in regs])
        if not m.any():
            continue
        ax.scatter(
            x[m],
            y[m],
            s=28,
            c=REGION_COLOR[reg],
            alpha=0.88,
            edgecolors="0.12",
            linewidths=0.35,
            label=reg,
            zorder=3,
        )
    # mark remaining failures
    mfail = np.asarray(failed, dtype=bool)
    if mfail.any():
        ax.scatter(
            x[mfail],
            y[mfail],
            s=90,
            facecolors="none",
            edgecolors="white",
            linewidths=2.4,
            zorder=4,
        )
        ax.scatter(
            x[mfail],
            y[mfail],
            s=90,
            facecolors="none",
            edgecolors="0.08",
            linewidths=1.25,
            zorder=5,
            label="failed after R2",
        )
    lim = [0.25, 1.02]
    ax.plot(lim, lim, "--", color="0.5", lw=0.9, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("maxnet AUC")
    ax.set_ylabel("Standard KAN e2e AUC (ITT)")
    ax.set_aspect("equal", adjustable="box")
    mu, lo, hi = _boot_mean_ci(y - x)
    ax.set_title(
        f"Standard KAN end-to-end (ITT) vs maxnet\n"
        f"mean ΔAUC (e2e − maxnet) = {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]",
        fontsize=9,
    )
    ax.legend(frameon=False, loc="upper left", fontsize=7, ncol=2)
    _despine(ax)
    _save(fig, OUT_MS, "Fig1_standard_kan_e2e_maxnet")


def plot_fig2_auc_box(wide: pd.DataFrame) -> None:
    models = [
        ("additive_kan_ipp", "B-spline IPP", COLOR_ADDITIVE),
        ("gam_ipp_same_basis", "GAM-IPP", COLOR_GAM),
        ("maxnet_bg10k", "maxnet", COLOR_MAXNET),
    ]
    data, labels, colors = [], [], []
    for col, lab, c in models:
        if col not in wide.columns:
            continue
        y = wide[col].values.astype(float)
        y = y[np.isfinite(y)]
        data.append(y)
        labels.append(lab)
        colors.append(c)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="0.15", lw=1.2),
        whiskerprops=dict(color="0.3", lw=0.9),
        capprops=dict(color="0.3", lw=0.9),
        boxprops=dict(lw=0.9, edgecolor="0.25"),
    )
    ax.set_xticklabels(labels, rotation=12)
    for patch, fc in zip(bp["boxes"], colors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.95)
    rng = np.random.default_rng(0)
    for i, y in enumerate(data, start=1):
        ax.plot(i + rng.uniform(-0.12, 0.12, size=len(y)), y, "o", ms=2.2, color="0.25", alpha=0.14, mew=0)
        ax.plot(i, np.mean(y), "D", color="0.1", ms=5, zorder=4)
    ax.set_ylabel("AUC")
    ax.set_title(f"A  AUC distribution (n = {len(data[0])})")
    ax.set_ylim(0.28, 1.02)
    _annotate_boxplot_medians(ax, data, fontsize=7.0, face_colors=colors, placement="median")
    _despine(ax)

    ax = axes[1]
    means = [np.mean(y) for y in data]
    ses = [np.std(y, ddof=1) / np.sqrt(len(y)) for y in data]
    order = np.argsort(means)
    ypos = np.arange(len(order))
    mean_ord = [means[i] for i in order]
    ax.barh(
        ypos,
        mean_ord,
        xerr=[ses[i] for i in order],
        color=[colors[i] for i in order],
        edgecolor="0.2",
        height=0.6,
        alpha=0.95,
        error_kw=dict(ecolor="0.2", lw=0.8, capsize=2),
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlabel("Mean AUC ± SE")
    ax.set_title("B  Method means")
    ax.set_xlim(0.55, 0.78)
    _annotate_bar_values(ax, mean_ord, positions=ypos, orient="h", fontsize=7.0)
    _despine(ax)
    fig.tight_layout()
    _save(fig, OUT_MS, "Fig2_species_AUC_boxplots")


def plot_fig3_paired(wide: pd.DataFrame, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.2))

    # A Add vs GAM
    ax = axes[0, 0]
    x = wide["additive_kan_ipp"].values
    y = wide["gam_ipp_same_basis"].values
    regs = wide["region"].values
    for reg in REGION_ORDER:
        m = regs == reg
        ax.scatter(
            x[m],
            y[m],
            s=20,
            c=REGION_COLOR[reg],
            alpha=0.85,
            edgecolors="0.15",
            linewidths=0.25,
            label=reg,
        )
    lim = [0.3, 1.02]
    ax.plot(lim, lim, "--", color="0.5", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("B-spline IPP AUC")
    ax.set_ylabel("GAM-IPP AUC")
    ax.set_title("A  Same-basis GAM")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    _despine(ax)

    # B Add vs maxnet
    ax = axes[0, 1]
    y = wide["maxnet_bg10k"].values
    for reg in REGION_ORDER:
        m = regs == reg
        ax.scatter(
            x[m],
            y[m],
            s=20,
            c=REGION_COLOR[reg],
            alpha=0.85,
            edgecolors="0.15",
            linewidths=0.25,
        )
    ax.plot(lim, lim, "--", color="0.5", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("B-spline IPP AUC")
    ax.set_ylabel("maxnet AUC")
    d = x - y
    mu, lo, hi = _boot_mean_ci(d)
    ax.set_title(f"B  maxnet  (Δ={mu:+.3f})")
    _despine(ax)

    # C delta distributions (additive only; residual depth is Fig.5)
    ax = axes[1, 0]
    d_gam = wide["additive_kan_ipp"] - wide["gam_ipp_same_basis"]
    d_mx = wide["additive_kan_ipp"] - wide["maxnet_bg10k"]
    data = [d_gam.values, d_mx.values]
    labs = ["Add−GAM", "Add−maxnet"]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    ax.set_xticklabels(labs, rotation=0)
    for patch, c in zip(bp["boxes"], [COLOR_GAM, COLOR_MAXNET]):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax.axhline(0, color="0.4", ls="--", lw=0.8)
    ax.set_ylabel("ΔAUC")
    ax.set_title("C  Paired contrasts")
    _annotate_boxplot_medians(
        ax, data, fontsize=7.0, fmt="{:+.3f}",
        face_colors=[COLOR_GAM, COLOR_MAXNET],
    )
    _despine(ax)

    # D regional Add-maxnet
    ax = axes[1, 1]
    deltas, labels, cols = [], [], []
    for reg in REGION_ORDER:
        w = wide[wide.region == reg]
        d = (w["additive_kan_ipp"] - w["maxnet_bg10k"]).values
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        deltas.append(d)
        labels.append(reg)
        cols.append(REGION_COLOR[reg])
    bp = ax.boxplot(deltas, patch_artist=True, widths=0.55, showfliers=False)
    ax.set_xticklabels(labels)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)
    ax.axhline(0, color="0.4", ls="--", lw=0.8)
    ax.set_ylabel("ΔAUC (Add − maxnet)")
    ax.set_title("D  By region")
    _annotate_boxplot_medians(ax, deltas, fontsize=6.5, fmt="{:+.3f}", face_colors=cols)
    _despine(ax)

    fig.tight_layout()
    _save(fig, OUT_MS, "Fig3_paired_comparisons")


def plot_fig4_ranking(wide: pd.DataFrame) -> None:
    """Left–right layout matching original plot_ms_all.py (not stacked)."""
    # rank by Additive AUC (ascending), like original
    kan_sorted = wide.set_index(["region", "species_id"])["additive_kan_ipp"].sort_values()
    idx = np.arange(len(kan_sorted))
    max_vals = wide.set_index(["region", "species_id"]).loc[kan_sorted.index, "maxnet_bg10k"].values
    kan_vals = kan_sorted.values

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))

    ax = axes[0]
    ax.fill_between(
        idx,
        kan_vals,
        max_vals,
        where=kan_vals >= max_vals,
        color=COLOR_RANK_FILL_IPP,
        alpha=0.35,
        label="B-spline IPP > maxnet",
    )
    ax.fill_between(
        idx,
        kan_vals,
        max_vals,
        where=kan_vals < max_vals,
        color=COLOR_RANK_FILL_MAX,
        alpha=0.35,
        label="maxnet > B-spline IPP",
    )
    ax.plot(idx, kan_vals, color=COLOR_RANK_IPP, lw=1.15, label="B-spline IPP")
    ax.plot(idx, max_vals, color=COLOR_RANK_MAX, lw=1.15, label="maxnet")
    ax.set_xlabel("Species rank (by B-spline IPP AUC)")
    ax.set_ylabel("AUC")
    ax.set_title(f"A  B-spline IPP vs maxnet (n = {len(kan_sorted)})")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    _despine(ax)

    ax = axes[1]
    delta = (
        wide.set_index(["region", "species_id"])["additive_kan_ipp"]
        - wide.set_index(["region", "species_id"])["maxnet_bg10k"]
    )
    delta_sorted = delta.sort_values()
    y_pos = np.arange(len(delta_sorted))
    colors_d = [REGION_COLOR[r] for r, _ in delta_sorted.index]
    ax.barh(y_pos, delta_sorted.values, height=0.85, color=colors_d, edgecolor="none", alpha=0.9)
    ax.axvline(0, color="0.35", ls="--", lw=0.9, zorder=2)
    ax.set_xlabel(r"$\Delta$AUC (B-spline IPP $-$ maxnet)")
    ax.set_title("B  Per-species difference")
    ax.set_yticks([])
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=REGION_COLOR[r], edgecolor="none")
        for r in REGION_ORDER
    ]
    ax.legend(
        handles,
        REGION_ORDER,
        fontsize=7,
        loc="lower right",
        ncol=3,
        title="Region",
        title_fontsize=7,
        frameon=False,
    )
    _despine(ax)
    fig.tight_layout()
    _save(fig, OUT_MS, "Fig4_species_ranking")


def plot_fig5_npo(wide: pd.DataFrame) -> None:
    """Restore smooth trend curves (UnivariateSpline on log10 n_PO), region-colored points."""
    from scipy.interpolate import UnivariateSpline

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    n = wide["n_presence"].astype(float)
    auc = wide["additive_kan_ipp"].astype(float)
    dlt = (wide["additive_kan_ipp"] - wide["maxnet_bg10k"]).astype(float)
    reg = wide["region"]

    ax = axes[0]
    for r in REGION_ORDER:
        m = (reg == r) & np.isfinite(n) & np.isfinite(auc) & (n > 0)
        ax.scatter(
            n[m],
            auc[m],
            s=16,
            c=REGION_COLOR[r],
            alpha=0.75,
            edgecolors="0.15",
            linewidths=0.25,
            zorder=2,
            label=r,
        )
    valid = np.isfinite(n) & np.isfinite(auc) & (n > 0)
    order_n = np.argsort(n[valid].values)
    x_raw = n[valid].values[order_n]
    y_raw = auc[valid].values[order_n]
    try:
        spl = UnivariateSpline(np.log10(x_raw + 1), y_raw, s=max(len(order_n) * 0.04, 1.0))
        xc = np.logspace(np.log10(x_raw.min() + 1), np.log10(x_raw.max() + 1), 120)
        ax.plot(xc, spl(np.log10(xc + 1)), color="0.12", lw=1.9, zorder=3)
    except Exception as e:
        print("Fig5A spline failed:", e)
    ax.set_xscale("log")
    ax.set_xlabel("Training presence records (n)")
    ax.set_ylabel("AUC (B-spline IPP)")
    ax.set_title("A  Discrimination vs sample size")
    _despine(ax)

    ax = axes[1]
    ax.axhline(0, color="0.35", ls="--", lw=0.9, zorder=0)
    for r in REGION_ORDER:
        m = (reg == r) & np.isfinite(n) & np.isfinite(dlt) & (n > 0)
        ax.scatter(
            n[m],
            dlt[m],
            s=16,
            c=REGION_COLOR[r],
            alpha=0.75,
            edgecolors="0.15",
            linewidths=0.25,
            zorder=2,
        )
    v2 = np.isfinite(n) & np.isfinite(dlt) & (n > 0)
    order2 = np.argsort(n[v2].values)
    x2 = n[v2].values[order2]
    y2 = dlt[v2].values[order2]
    try:
        spl2 = UnivariateSpline(np.log10(x2 + 1), y2, s=max(order2.shape[0] * 0.06, 1.0))
        xc2 = np.logspace(np.log10(x2.min() + 1), np.log10(x2.max() + 1), 120)
        ax.plot(xc2, spl2(np.log10(xc2 + 1)), color="0.12", lw=1.9, zorder=3)
    except Exception as e:
        print("Fig5B spline failed:", e)
    ax.set_xscale("log")
    ax.set_xlabel("Training presence records (n)")
    ax.set_ylabel(r"$\Delta$AUC (B-spline IPP $-$ maxnet)")
    ax.set_title("B  Model difference vs sample size")
    handles = [
        Line2D([0], [0], marker="o", color=REGION_COLOR[r], markersize=6, lw=0, label=r)
        for r in REGION_ORDER
    ]
    ax.legend(
        handles=handles,
        fontsize=7.5,
        loc="lower left",
        ncol=2,
        title="Region",
        title_fontsize=8,
        frameon=False,
    )
    _despine(ax)
    fig.tight_layout()
    _save(fig, OUT_MS, "Fig5_nPO_effect")


def plot_fig6_curves_copy_or_rebuild() -> None:
    """Prefer reusing phase5 curve panels if present; else skip composite."""
    # copy composite from v4 if exists
    src = ROOT / "figures" / "ms_results" / "v4" / "Fig6.png"
    OUT_MS.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        import shutil

        shutil.copy2(src, OUT_MS / "Fig6.png")
        # also copy component panels if useful
        for name in [
            "Fig6_A_ecdf",
            "Fig6_B_bars",
            "Fig6_C1",
            "Fig6_C2",
            "Fig6_C3",
            "Fig6_C4",
            "Fig6_C5",
            "Fig6_C6",
        ]:
            for ext in ("png", "pdf"):
                p = ROOT / "figures" / "ms_results" / "v4" / f"{name}.{ext}"
                if p.is_file():
                    shutil.copy2(p, OUT_MS / f"{name}.{ext}")
        print("copied Fig6 from v4 (same-basis curve check)")
    else:
        print("WARN: no Fig6 source; skipped")


def plot_fig7_deep(df: pd.DataFrame, wide: pd.DataFrame) -> None:
    add = wide.set_index(["region", "species_id"])["additive_kan_ipp"]
    arms = [
        ("B", "deep2_rphi", "Deep-2 Rφ", COLOR_DEEP2_RPHI),
        ("B", "deep2_rx", "Deep-2 Rx", COLOR_DEEP2_RX),
        ("C", "deep3_rphi", "Deep-3 Rφ", COLOR_DEEP3_RPHI),
        ("C", "deep3_rx", "Deep-3 Rx", COLOR_DEEP3_RX),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, stage_models, title in [
        (axes[0], arms[:2], "A  Deep-2 − Additive"),
        (axes[1], arms[2:], "B  Deep-3 − Additive"),
    ]:
        positions = []
        data = []
        colors = []
        tick_pos = []
        tick_lab = []
        pos = 1.0
        for reg in REGION_ORDER:
            for stage, model, lab, col in stage_models:
                s = deep_species_mean(df, stage, model)
                common = s.index.intersection(add.index)
                d = (s.loc[common] - add.loc[common]).reset_index()
                d.columns = ["region", "species_id", "delta"]
                dd = d.loc[d.region == reg, "delta"].values
                dd = dd[np.isfinite(dd)]
                data.append(dd)
                colors.append(col)
                positions.append(pos)
                pos += 0.9
            tick_pos.append(pos - 1.35)
            tick_lab.append(reg)
            pos += 0.55
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.7,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.88)
        ax.axhline(0, color="0.35", ls="--", lw=0.8)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
        ax.set_title(title)
        ax.set_ylabel("ΔAUC" if ax is axes[0] else "")
        # global means
        for stage, model, lab, col in stage_models:
            s = deep_species_mean(df, stage, model)
            common = s.index.intersection(add.index)
            mu, lo, hi = _boot_mean_ci((s.loc[common] - add.loc[common]).values)
            ax.plot([], [], color=col, lw=6, label=f"{lab} {mu:+.3f}")
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        _despine(ax)
    fig.tight_layout()
    _save(fig, OUT_MS, "Fig7_deepkan_ablation")


def plot_si_figures(df: pd.DataFrame, wide: pd.DataFrame) -> None:
    _setup_style()
    # S1 regional boxplots
    models = [
        ("additive_kan_ipp", "IPP", COLOR_ADDITIVE),
        ("gam_ipp_same_basis", "GAM", COLOR_GAM),
        ("maxnet_bg10k", "maxnet", COLOR_MAXNET),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6), sharey=True)
    for ax, reg in zip(axes.ravel(), REGION_ORDER):
        w = wide[wide.region == reg]
        data, cols = [], []
        for col, _, c in models:
            y = w[col].values.astype(float)
            data.append(y[np.isfinite(y)])
            cols.append(c)
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.9)
        ax.set_xticklabels([m[1] for m in models], rotation=15)
        ax.set_title(f"{reg} (n={len(w)})")
        ax.set_ylim(0.28, 1.02)
        if ax in axes[:, 0]:
            ax.set_ylabel("AUC")
        _annotate_boxplot_medians(
            ax, data, fontsize=6.5, face_colors=cols, placement="median"
        )
        _despine(ax)
    # No figure title in-image (captions live in SI.md).
    fig.tight_layout()
    _save(fig, OUT_SI, "FigS2_AUC_by_region")

    # S2 heatmaps — 2×3 panels + vertical colorbar on the far right (not between panels)
    fig = plt.figure(figsize=(10.4, 6.8))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        wspace=0.28,
        hspace=0.28,
        left=0.06,
        right=0.94,
        top=0.94,
        bottom=0.08,
    )
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)])
    im = None
    for ax, reg in zip(axes.ravel(), REGION_ORDER):
        w = wide[wide.region == reg].sort_values("additive_kan_ipp", ascending=False)
        mat = w[["additive_kan_ipp", "gam_ipp_same_basis", "maxnet_bg10k"]].values
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.4, vmax=0.98)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["IPP", "GAM", "maxnet"], fontsize=7)
        ax.set_yticks([])
        ax.set_title(reg)
    cax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("AUC")
    _save(fig, OUT_SI, "FigS3_heatmap_species_model_AUC")

    # S3 extremes
    wide = wide.copy()
    wide["d"] = wide["additive_kan_ipp"] - wide["maxnet_bg10k"]
    lo = wide.nsmallest(12, "d")
    hi = wide.nlargest(12, "d")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharex=False)
    for ax, sub, title in [
        (axes[0], lo.sort_values("d"), "A  Most negative Δ"),
        (axes[1], hi.sort_values("d"), "B  Most positive Δ"),
    ]:
        labs = [f"{r}/{s}" for r, s in zip(sub.region, sub.species_id)]
        cols = [REGION_COLOR[r] for r in sub.region]
        ypos = np.arange(len(sub))
        vals = sub["d"].values
        ax.barh(ypos, vals, color=cols, edgecolor="0.2")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labs, fontsize=7)
        ax.axvline(0, color="0.3", ls="--", lw=0.8)
        ax.set_xlabel("ΔAUC (Add − maxnet)")
        ax.set_title(title)
        _annotate_bar_values(
            ax, vals, positions=ypos, orient="h", fontsize=6.0, fmt="{:+.3f}", offset_frac=0.02
        )
        _despine(ax)
    fig.tight_layout()
    _save(fig, OUT_SI, "FigS4_extreme_delta_IPP_maxnet")

    # S4 metrics plane (AUC vs COR / AUPRC) from stage A means (main-analysis species)
    a = df[df.stage == "A"]
    if "lambda_selection" in a.columns:
        a = a[a.lambda_selection != "skipped_n_po_lt_5"]
    rows = []
    for model, lab, c in [
        ("additive_kan_ipp", "B-spline IPP", COLOR_ADDITIVE),
        ("gam_ipp_same_basis", "GAM-IPP", COLOR_GAM),
        ("maxnet_bg10k", "maxnet", COLOR_MAXNET),
    ]:
        sub = a[a.model == model]
        rows.append(
            {
                "lab": lab,
                "c": c,
                "auc": sub.auc_roc.mean(),
                "auc_se": sub.auc_roc.std(ddof=1) / np.sqrt(sub.auc_roc.notna().sum()),
                "cor": sub.cor.mean() if "cor" in sub else np.nan,
                "cor_se": sub.cor.std(ddof=1) / np.sqrt(sub.cor.notna().sum()) if "cor" in sub else np.nan,
                "auprc": sub.auprc.mean() if "auprc" in sub else np.nan,
                "auprc_se": sub.auprc.std(ddof=1) / np.sqrt(sub.auprc.notna().sum()) if "auprc" in sub else np.nan,
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, xk, yk, xl, yl, title in [
        (axes[0], "cor", "auc", "mean COR", "mean AUC", "A  AUC vs COR"),
        (axes[1], "auprc", "auc", "mean AUPRC", "mean AUC", "B  AUC vs AUPRC"),
    ]:
        for r in rows:
            ax.errorbar(
                r[xk],
                r[yk],
                xerr=r[f"{xk}_se"],
                yerr=r[f"{yk}_se"] if yk != "auc" else r["auc_se"],
                fmt="o",
                color=r["c"],
                ms=7,
                capsize=2,
                label=r["lab"],
            )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        _despine(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, OUT_SI, "FigS5_metrics_plane")

    # S_e2e: e2e - maxnet by region
    e2e = e2e_itt(df)
    mx = df[(df.stage == "A") & (df.model == "maxnet_bg10k")][
        ["region", "species_id", "auc_roc"]
    ].rename(columns={"auc_roc": "auc_mx"})
    mrg = e2e.merge(mx, on=["region", "species_id"], how="inner")
    mrg["delta"] = mrg["auc_roc"] - mrg["auc_mx"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    data, cols = [], []
    for reg in REGION_ORDER:
        d = mrg.loc[mrg.region == reg, "delta"].values
        d = d[np.isfinite(d)]
        data.append(d)
        cols.append(REGION_COLOR[reg])
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    ax.set_xticklabels(REGION_ORDER)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)
    ax.axhline(0, color="0.35", ls="--", lw=0.8)
    mu, lo, hi = _boot_mean_ci(mrg["delta"].values)
    ax.set_title(f"e2e − maxnet by region  (global {mu:+.3f} [{lo:+.3f},{hi:+.3f}])")
    ax.set_ylabel("ΔAUC")
    _annotate_boxplot_medians(ax, data, fontsize=7.0, fmt="{:+.3f}", face_colors=cols)
    _despine(ax)
    _save(fig, OUT_SI, "FigS1_e2e_minus_maxnet_by_region")


def write_si_tables(df: pd.DataFrame, wide: pd.DataFrame) -> None:
    """Write canonical SI table CSVs under SI/tables/ (single numbering scheme).

    Does not overwrite the polished narrative SI/SI.md.
    """
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    if "lambda_selection" in df.columns:
        a_mask = (df.stage == "A") & (df.lambda_selection != "skipped_n_po_lt_5")
    else:
        a_mask = df.stage == "A"
    df_a = df[a_mask]
    if "lambda_selection" in wide.columns:
        wide = wide[wide.lambda_selection != "skipped_n_po_lt_5"].copy()

    rows = []
    for model, lab in [
        ("additive_kan_ipp", "B-spline IPP"),
        ("gam_ipp_same_basis", "same-basis GAM-IPP"),
        ("maxnet_bg10k", "maxnet (bg ≤ 10,000)"),
        ("maxnet_bg50k", "maxnet (bg ≤ 50,000)"),
    ]:
        a = df_a[df_a.model == model]
        a = a[np.isfinite(a.auc_roc)]
        rows.append(
            {
                "model": lab,
                "model_id": model,
                "n": int(a.species_id.nunique()) if "species_id" in a else len(a),
                "AUC_mean": a.auc_roc.mean(),
                "AUC_sd": a.auc_roc.std(ddof=1),
                "AUPRC_mean": a.auprc.mean() if "auprc" in a else np.nan,
                "AUPRC_sd": a.auprc.std(ddof=1) if "auprc" in a else np.nan,
                "PRG_mean": a.prg.mean() if "prg" in a else np.nan,
                "PRG_sd": a.prg.std(ddof=1) if "prg" in a else np.nan,
                "COR_mean": a.cor.mean() if "cor" in a else np.nan,
                "COR_sd": a.cor.std(ddof=1) if "cor" in a else np.nan,
            }
        )
    add = wide.set_index(["region", "species_id"])["additive_kan_ipp"]
    for stage, model, lab in [
        ("B", "deep2_rphi", "Deep-2 (Rφ)"),
        ("B", "deep2_rx", "Deep-2 (Rx)"),
        ("C", "deep3_rphi", "Deep-3 (Rφ)"),
        ("C", "deep3_rx", "Deep-3 (Rx)"),
    ]:
        s = deep_species_mean(df, stage, model)
        common = s.index.intersection(add.index)
        rows.append(
            {
                "model": lab,
                "model_id": model,
                "n": len(common),
                "AUC_mean": s.loc[common].mean(),
                "AUC_sd": s.loc[common].std(ddof=1),
                "AUPRC_mean": np.nan,
                "AUPRC_sd": np.nan,
                "PRG_mean": np.nan,
                "PRG_sd": np.nan,
                "COR_mean": np.nan,
                "COR_sd": np.nan,
            }
        )
    e2e = e2e_itt(df).set_index(["region", "species_id"])
    common = e2e.index.intersection(add.index)
    e2e_c = e2e.loc[common]
    rows.append(
        {
            "model": "end-to-end standard KAN",
            "model_id": "standard_kan_ipp_itt",
            "n": int(e2e_c.auc_roc.notna().sum()),
            "AUC_mean": e2e_c.auc_roc.mean(),
            "AUC_sd": e2e_c.auc_roc.std(ddof=1),
            "AUPRC_mean": e2e_c.auprc.mean() if "auprc" in e2e_c else np.nan,
            "AUPRC_sd": e2e_c.auprc.std(ddof=1) if "auprc" in e2e_c else np.nan,
            "PRG_mean": e2e_c.prg.mean() if "prg" in e2e_c else np.nan,
            "PRG_sd": e2e_c.prg.std(ddof=1) if "prg" in e2e_c else np.nan,
            "COR_mean": e2e_c.cor.mean() if "cor" in e2e_c else np.nan,
            "COR_sd": e2e_c.cor.std(ddof=1) if "cor" in e2e_c else np.nan,
        }
    )
    pd.DataFrame(rows).to_csv(OUT_TAB / "TableS1_global_metrics.csv", index=False)

    reg_rows = []
    for reg in REGION_ORDER:
        w = wide[wide.region == reg]
        reg_rows.append(
            {
                "region": reg,
                "n": len(w),
                "AUC_add_mean": w.additive_kan_ipp.mean(),
                "AUC_add_sd": w.additive_kan_ipp.std(ddof=1),
                "AUC_gam_mean": w.gam_ipp_same_basis.mean(),
                "AUC_maxnet10_mean": w.maxnet_bg10k.mean(),
                "delta_add_minus_maxnet": (w.additive_kan_ipp - w.maxnet_bg10k).mean(),
            }
        )
    pd.DataFrame(reg_rows).to_csv(OUT_TAB / "TableS2_regional_AUC.csv", index=False)

    s3_rows, s4_rows = [], []
    for reg in REGION_ORDER:
        row3, row4 = {"region": reg}, {"region": reg}
        for model, col in [
            ("additive_kan_ipp", "B-spline IPP"),
            ("gam_ipp_same_basis", "same-basis GAM-IPP"),
            ("maxnet_bg10k", "maxnet"),
        ]:
            sub = df_a[(df_a.model == model) & (df_a.region == reg)]
            row3[col] = sub.auprc.mean() if len(sub) else np.nan
            row4[col] = sub.prg.mean() if len(sub) else np.nan
        s3_rows.append(row3)
        s4_rows.append(row4)
    pd.DataFrame(s3_rows).to_csv(OUT_TAB / "TableS3_regional_AUPRC.csv", index=False)
    pd.DataFrame(s4_rows).to_csv(OUT_TAB / "TableS4_regional_PRG.csv", index=False)

    pairs = []
    d = wide.additive_kan_ipp - wide.gam_ipp_same_basis
    mu, lo, hi = _boot_mean_ci(d.values)
    pairs.append(
        {
            "contrast": "B-spline IPP − same-basis GAM-IPP",
            "mean_delta_AUC": mu,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": int(d.notna().sum()),
        }
    )
    d = wide.additive_kan_ipp - wide.maxnet_bg10k
    mu, lo, hi = _boot_mean_ci(d.values)
    pairs.append(
        {
            "contrast": "B-spline IPP − maxnet (bg ≤ 10,000)",
            "mean_delta_AUC": mu,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": int(d.notna().sum()),
        }
    )
    for stage, model, lab in [
        ("B", "deep2_rphi", "Deep-2 (Rφ) − Additive"),
        ("B", "deep2_rx", "Deep-2 (Rx) − Additive"),
        ("C", "deep3_rphi", "Deep-3 (Rφ) − Additive"),
        ("C", "deep3_rx", "Deep-3 (Rx) − Additive"),
    ]:
        s = deep_species_mean(df, stage, model)
        add_s = wide.set_index(["region", "species_id"]).additive_kan_ipp
        common = s.index.intersection(add_s.index)
        dd = (s.loc[common] - add_s.loc[common]).values
        mu, lo, hi = _boot_mean_ci(dd)
        pairs.append(
            {"contrast": lab, "mean_delta_AUC": mu, "ci_lo": lo, "ci_hi": hi, "n": len(dd)}
        )
    e2e = e2e_itt(df).set_index(["region", "species_id"])
    mx = wide.set_index(["region", "species_id"]).maxnet_bg10k
    common = e2e.index.intersection(mx.index)
    dd = (e2e.loc[common, "auc_roc"] - mx.loc[common]).values
    mu, lo, hi = _boot_mean_ci(dd)
    pairs.append(
        {
            "contrast": "end-to-end standard KAN − maxnet",
            "mean_delta_AUC": mu,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": len(dd),
        }
    )
    pd.DataFrame(pairs).to_csv(OUT_TAB / "TableS5_paired_delta_AUC.csv", index=False)

    bins = [0, 10, 30, 100, 500, 10**9]
    labels = ["≤10", "11–30", "31–100", "101–500", ">500"]
    w = wide.copy()
    w["n_PO_bin"] = pd.cut(w.n_presence, bins=bins, labels=labels)
    s6 = (
        w.groupby("n_PO_bin", observed=False)
        .agg(
            n=("species_id", "count"),
            AUC_mean=("additive_kan_ipp", "mean"),
            AUC_sd=("additive_kan_ipp", lambda x: x.std(ddof=1)),
        )
        .reset_index()
    )
    s6.to_csv(OUT_TAB / "TableS6_nPO_bins_AUC.csv", index=False)

    can = wide[wide.region == "CAN"][
        ["species_id", "additive_kan_ipp", "gam_ipp_same_basis", "maxnet_bg10k"]
    ].copy()
    can = can.rename(
        columns={
            "species_id": "species",
            "additive_kan_ipp": "B-spline IPP",
            "gam_ipp_same_basis": "same-basis GAM-IPP",
            "maxnet_bg10k": "maxnet",
        }
    ).sort_values("species")
    can.to_csv(OUT_TAB / "TableS7_CAN_species_models.csv", index=False)

    deep_reg = []
    add_s = wide.set_index(["region", "species_id"]).additive_kan_ipp
    for stage, model, lab in [
        ("B", "deep2_rphi", "deep2_rphi"),
        ("B", "deep2_rx", "deep2_rx"),
        ("C", "deep3_rphi", "deep3_rphi"),
        ("C", "deep3_rx", "deep3_rx"),
    ]:
        s = deep_species_mean(df, stage, model)
        for reg in REGION_ORDER:
            idx = [i for i in s.index if i[0] == reg and i in add_s.index]
            if not idx:
                continue
            dd = (s.loc[idx] - add_s.loc[idx]).values
            mu, lo, hi = _boot_mean_ci(dd)
            deep_reg.append(
                {
                    "model": lab,
                    "region": reg,
                    "n": len(dd),
                    "mean_delta": mu,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    pd.DataFrame(deep_reg).to_csv(OUT_TAB / "TableS10_deep_regional_delta.csv", index=False)

    lam = wide[["region", "species_id", "n_presence", "lambda_s", "lambda_selection"]].drop_duplicates()
    lam.to_csv(OUT_TAB / "TableS13_lambda_star.csv", index=False)

    e2e = e2e_itt(df)
    mx = df_a[df_a.model == "maxnet_bg10k"][["region", "species_id", "auc_roc"]].rename(
        columns={"auc_roc": "auc_maxnet"}
    )
    e2e = e2e.merge(mx, on=["region", "species_id"], how="left")
    e2e["delta_e2e_minus_maxnet"] = e2e["auc_roc"] - e2e["auc_maxnet"]
    e2e.to_csv(OUT_TAB / "TableS14_e2e_species_itt.csv", index=False)

    mat = wide[
        [
            "region",
            "species_id",
            "n_presence",
            "lambda_s",
            "additive_kan_ipp",
            "gam_ipp_same_basis",
            "maxnet_bg10k",
        ]
    ].copy()
    for stage, model, col in [
        ("B", "deep2_rphi", "deep2_rphi"),
        ("B", "deep2_rx", "deep2_rx"),
        ("C", "deep3_rphi", "deep3_rphi"),
        ("C", "deep3_rx", "deep3_rx"),
    ]:
        s = deep_species_mean(df, stage, model).rename(col)
        mat = mat.merge(s.reset_index(), on=["region", "species_id"], how="left")
    e2e_s = e2e_itt(df)[["region", "species_id", "auc_roc", "remediation", "failed"]].rename(
        columns={"auc_roc": "e2e_auc"}
    )
    mat = mat.merge(e2e_s, on=["region", "species_id"], how="left")
    mat.to_csv(OUT_TAB / "TableS12_species_model_AUC.csv", index=False)

    cor_w = (
        df_a.pivot_table(
            index=["region", "species_id"], columns="model", values="cor", aggfunc="first"
        )
        .reset_index()
    )
    e2e_cor = e2e_itt(df)[["region", "species_id", "cor"]].rename(columns={"cor": "e2e_cor"})
    cor_w = cor_w.merge(e2e_cor, on=["region", "species_id"], how="left")
    cor_w.to_csv(OUT_TAB / "TableS12_species_model_COR.csv", index=False)

    print("wrote SI tables to", OUT_TAB)


def write_si_md() -> None:
    """Do not overwrite the polished SI/SI.md. Refresh English pointer only."""
    en = Path(ROOT / "SI" / "SI_en.md")
    en.write_text(
        "# Supporting Information (English pointer)\n\n"
        "Primary SI narrative is maintained in `SI/SI.md` (Chinese captions; short tables embedded).\n"
        "Machine-readable tables: `SI/tables/TableS1–S14` (Methodological Closure; single numbering).\n"
        "Figures: `SI/figures/FigS1–S5` (S1 = e2e − maxnet by region; S5 = multi-metrics plane).\n"
        "Legacy freeze-λ tables: `SI/tables/_archive_legacy/`.\n"
        "Regenerate figures/tables: `python benchmarks/plot_ms_closure_v7.2.py` "
        "(does not rewrite `SI.md` prose).\n",
        encoding="utf-8",
    )
    print("refreshed SI/SI_en.md (SI.md left unchanged)")


def _compose_pngs(panel_paths, out_name: str, nrows: int, ncols: int, figsize) -> None:
    """Stitch already-rendered panel images into one figure; letters stay as drawn."""
    from matplotlib import image as mpimg

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes_flat, panel_paths):
        ax.imshow(mpimg.imread(str(p)))
        ax.axis("off")
    for ax in axes_flat[len(panel_paths):]:
        ax.axis("off")
    fig.tight_layout(pad=0.15)
    _save(fig, OUT_MS, out_name)


def plot_fig2_performance_composite(wide: pd.DataFrame, df: pd.DataFrame) -> None:
    """§1.2 表现：分布/均值/配对/分区 A–F；不含 Deep（深度见 Fig.5）。"""
    models = [
        ("additive_kan_ipp", "B-spline IPP", COLOR_ADDITIVE),
        ("gam_ipp_same_basis", "GAM-IPP", COLOR_GAM),
        ("maxnet_bg10k", "maxnet", COLOR_MAXNET),
    ]
    data, labels, colors = [], [], []
    for col, lab, c in models:
        if col not in wide.columns:
            continue
        y = wide[col].values.astype(float)
        y = y[np.isfinite(y)]
        data.append(y)
        labels.append(lab)
        colors.append(c)

    fig = plt.figure(figsize=(8.8, 10.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.05, 1.05], hspace=0.40, wspace=0.30)

    # A box
    ax = fig.add_subplot(gs[0, 0])
    bp = ax.boxplot(
        data, patch_artist=True, widths=0.55, showfliers=False,
        medianprops=dict(color="0.15", lw=1.2),
        whiskerprops=dict(color="0.3", lw=0.9),
        capprops=dict(color="0.3", lw=0.9),
        boxprops=dict(lw=0.9, edgecolor="0.25"),
    )
    ax.set_xticklabels(labels, rotation=12)
    for patch, fc in zip(bp["boxes"], colors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.95)
    rng = np.random.default_rng(0)
    for i, y in enumerate(data, start=1):
        ax.plot(i + rng.uniform(-0.12, 0.12, size=len(y)), y, "o", ms=2.2, color="0.25", alpha=0.14, mew=0)
        ax.plot(i, np.mean(y), "D", color="0.1", ms=5, zorder=4)
    ax.set_ylabel("Independent-PA AUC")
    ax.set_title(
        f"A  Species-level AUC distributions\n"
        f"(B-spline IPP, same-basis GAM-IPP, maxnet; n = {len(data[0])})",
        fontsize=8.5,
    )
    ax.set_ylim(0.28, 1.02)
    _annotate_boxplot_medians(ax, data, fontsize=7.0, face_colors=colors, placement="median")
    _despine(ax)

    # B means
    ax = fig.add_subplot(gs[0, 1])
    means = [np.mean(y) for y in data]
    ses = [np.std(y, ddof=1) / np.sqrt(len(y)) for y in data]
    order = np.argsort(means)
    ypos = np.arange(len(order))
    mean_ord = [means[i] for i in order]
    ax.barh(
        ypos, mean_ord, xerr=[ses[i] for i in order],
        color=[colors[i] for i in order], edgecolor="0.2", height=0.6, alpha=0.95,
        error_kw=dict(ecolor="0.2", lw=0.8, capsize=2),
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlabel("Mean independent-PA AUC ± SE across species")
    ax.set_title(
        "B  Mean AUC by method\n(± SE across species)",
        fontsize=8.5,
    )
    ax.set_xlim(0.55, 0.82)
    se_ord = [ses[i] for i in order]
    for y0, mu, se in zip(ypos, mean_ord, se_ord):
        ax.text(
            mu + se + 0.008,
            y0,
            f"{mu:.3f}",
            ha="left",
            va="center",
            fontsize=7.0,
            color="0.15",
            clip_on=False,
            zorder=6,
        )
    _despine(ax)

    # C–F paired (additive only; residual depth is Fig.5)
    x = wide["additive_kan_ipp"].values
    regs = wide["region"].values
    lim = [0.3, 1.02]

    ax = fig.add_subplot(gs[1, 0])
    y = wide["gam_ipp_same_basis"].values
    for reg in REGION_ORDER:
        m = regs == reg
        ax.scatter(x[m], y[m], s=18, c=REGION_COLOR[reg], alpha=0.85, edgecolors="0.15", linewidths=0.25, label=reg)
    ax.plot(lim, lim, "--", color="0.5", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("B-spline IPP AUC"); ax.set_ylabel("Same-basis GAM-IPP AUC")
    ax.set_title(
        "C  B-spline IPP vs same-basis GAM-IPP\n(species as paired units; 1:1 line)",
        fontsize=8.5,
    )
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    _despine(ax)

    ax = fig.add_subplot(gs[1, 1])
    y = wide["maxnet_bg10k"].values
    for reg in REGION_ORDER:
        m = regs == reg
        ax.scatter(x[m], y[m], s=18, c=REGION_COLOR[reg], alpha=0.85, edgecolors="0.15", linewidths=0.25)
    ax.plot(lim, lim, "--", color="0.5", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("B-spline IPP AUC"); ax.set_ylabel("maxnet (bg ≤ 10,000) AUC")
    mu, lo, hi = _boot_mean_ci(x - y)
    ax.set_title(
        f"D  B-spline IPP vs maxnet@10k\n"
        f"mean ΔAUC (IPP − maxnet) = {mu:+.3f}",
        fontsize=8.5,
    )
    _despine(ax)

    ax = fig.add_subplot(gs[2, 0])
    d_gam = wide["additive_kan_ipp"] - wide["gam_ipp_same_basis"]
    d_mx = wide["additive_kan_ipp"] - wide["maxnet_bg10k"]
    d_e = [d_gam.values, d_mx.values]
    bp = ax.boxplot(d_e, patch_artist=True, widths=0.55, showfliers=False)
    ax.set_xticklabels(
        ["B-spline IPP − GAM-IPP", "B-spline IPP − maxnet"],
        rotation=12,
        fontsize=7.5,
    )
    for patch, c in zip(bp["boxes"], [COLOR_GAM, COLOR_MAXNET]):
        patch.set_facecolor(c)
        patch.set_alpha(0.88)
        patch.set_edgecolor("0.2")
    ax.axhline(0, color="0.4", ls="--", lw=0.8)
    ax.set_ylabel("Paired ΔAUC")
    ax.set_title(
        "E  Distribution of paired ΔAUC\n(additive contrasts only; no residual Deep)",
        fontsize=8.5,
    )
    _annotate_boxplot_medians(
        ax, d_e, fontsize=7.0, fmt="{:+.3f}",
        face_colors=[COLOR_GAM, COLOR_MAXNET],
    )
    handles_e = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_GAM, edgecolor="0.2", label="IPP − GAM"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAXNET, edgecolor="0.2", label="IPP − maxnet"),
    ]
    ax.legend(handles=handles_e, frameon=False, fontsize=7, loc="upper left")
    _despine(ax)

    ax = fig.add_subplot(gs[2, 1])
    deltas, labs, cols = [], [], []
    for reg in REGION_ORDER:
        w = wide[wide.region == reg]
        d = (w["additive_kan_ipp"] - w["maxnet_bg10k"]).values
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        deltas.append(d); labs.append(reg); cols.append(REGION_COLOR[reg])
    bp = ax.boxplot(deltas, patch_artist=True, widths=0.55, showfliers=False)
    ax.set_xticklabels(labs)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.9)
    ax.axhline(0, color="0.4", ls="--", lw=0.8)
    ax.set_ylabel("ΔAUC (B-spline IPP − maxnet)")
    ax.set_title(
        "F  B-spline IPP − maxnet ΔAUC by region\n(species as units)",
        fontsize=8.5,
    )
    _annotate_boxplot_medians(ax, deltas, fontsize=6.5, fmt="{:+.3f}", face_colors=cols)
    _despine(ax)

    _save(fig, OUT_MS, "Fig2_additive_performance")


def plot_fig3_robustness_composite(wide: pd.DataFrame) -> None:
    """§1.3 稳健：排序 + n_PO，A–D。字号刻意加大（阅读反馈：此前偏小）。"""
    from scipy.interpolate import UnivariateSpline

    # Local type scale: 2×2 layout fixed; bump further for on-screen readability
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEG = 14.0, 13.0, 12.0, 11.0

    kan_sorted = wide.set_index(["region", "species_id"])["additive_kan_ipp"].sort_values()
    idx = np.arange(len(kan_sorted))
    max_vals = wide.set_index(["region", "species_id"]).loc[kan_sorted.index, "maxnet_bg10k"].values
    kan_vals = kan_sorted.values

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.2))

    ax = axes[0, 0]
    ax.fill_between(idx, kan_vals, max_vals, where=kan_vals >= max_vals, color=COLOR_RANK_FILL_IPP, alpha=0.35, label="B-spline IPP > maxnet")
    ax.fill_between(idx, kan_vals, max_vals, where=kan_vals < max_vals, color=COLOR_RANK_FILL_MAX, alpha=0.35, label="maxnet > B-spline IPP")
    ax.plot(idx, kan_vals, color=COLOR_RANK_IPP, lw=1.15, label="B-spline IPP")
    ax.plot(idx, max_vals, color=COLOR_RANK_MAX, lw=1.15, label="maxnet")
    ax.set_xlabel("Species rank (ascending by B-spline IPP AUC)", fontsize=FS_LABEL)
    ax.set_ylabel("Independent-PA AUC", fontsize=FS_LABEL)
    ax.set_title(
        f"A  Ranking consistency: B-spline IPP and maxnet\n"
        f"(species ordered by B-spline IPP AUC; n = {len(kan_sorted)})",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG, loc="lower right", frameon=False)
    _despine(ax)

    ax = axes[0, 1]
    delta = (
        wide.set_index(["region", "species_id"])["additive_kan_ipp"]
        - wide.set_index(["region", "species_id"])["maxnet_bg10k"]
    ).sort_values()
    y_pos = np.arange(len(delta))
    colors_d = [REGION_COLOR[r] for r, _ in delta.index]
    ax.barh(y_pos, delta.values, height=0.85, color=colors_d, edgecolor="none", alpha=0.9)
    ax.axvline(0, color="0.35", ls="--", lw=0.9, zorder=2)
    ax.set_xlabel(r"$\Delta$AUC (B-spline IPP $-$ maxnet)", fontsize=FS_LABEL)
    ax.set_title(
        "B  Per-species ΔAUC (B-spline IPP − maxnet)\n"
        "(bars sorted; colours = NCEAS region)",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=FS_TICK)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=REGION_COLOR[r], edgecolor="none") for r in REGION_ORDER]
    # 6 regions → 3 rows × 2 columns; lower-right to clear the bar field
    ax.legend(
        handles, REGION_ORDER, fontsize=FS_LEG, loc="lower right", ncol=2,
        title="Region", title_fontsize=FS_LEG, frameon=False,
    )
    _despine(ax)

    n = wide["n_presence"].astype(float)
    auc = wide["additive_kan_ipp"].astype(float)
    dlt = (wide["additive_kan_ipp"] - wide["maxnet_bg10k"]).astype(float)
    reg = wide["region"]

    ax = axes[1, 0]
    for r in REGION_ORDER:
        m = (reg == r) & np.isfinite(n) & np.isfinite(auc) & (n > 0)
        ax.scatter(n[m], auc[m], s=28, c=REGION_COLOR[r], alpha=0.88, edgecolors="0.15", linewidths=0.3, zorder=2, label=r)
    valid = np.isfinite(n) & np.isfinite(auc) & (n > 0)
    order_n = np.argsort(n[valid].values)
    x_raw = n[valid].values[order_n]; y_raw = auc[valid].values[order_n]
    try:
        spl = UnivariateSpline(np.log10(x_raw + 1), y_raw, s=max(len(order_n) * 0.04, 1.0))
        xc = np.logspace(np.log10(x_raw.min() + 1), np.log10(x_raw.max() + 1), 120)
        ax.plot(xc, spl(np.log10(xc + 1)), color="0.12", lw=2.0, zorder=3)
    except Exception as e:
        print("Fig3C spline failed:", e)
    ax.set_xscale("log")
    ax.set_xlabel(r"Training presence-only sample size $n_{\mathrm{PO}}$ (log scale)", fontsize=FS_LABEL)
    ax.set_ylabel("B-spline IPP independent-PA AUC", fontsize=FS_LABEL)
    ax.set_title(
        "C  B-spline IPP AUC versus training sample size\n"
        "(points = species; black line = spline smoother)",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=REGION_COLOR[r], markersize=7, lw=0, label=r)
            for r in REGION_ORDER
        ],
        frameon=False,
        fontsize=FS_LEG,
        loc="lower right",
        ncol=2,
        title="Region",
        title_fontsize=FS_LEG,
    )
    _despine(ax)

    ax = axes[1, 1]
    ax.axhline(0, color="0.35", ls="--", lw=0.9, zorder=0)
    for r in REGION_ORDER:
        m = (reg == r) & np.isfinite(n) & np.isfinite(dlt) & (n > 0)
        ax.scatter(n[m], dlt[m], s=28, c=REGION_COLOR[r], alpha=0.88, edgecolors="0.15", linewidths=0.3, zorder=2)
    v2 = np.isfinite(n) & np.isfinite(dlt) & (n > 0)
    order2 = np.argsort(n[v2].values)
    x2 = n[v2].values[order2]; y2 = dlt[v2].values[order2]
    try:
        spl2 = UnivariateSpline(np.log10(x2 + 1), y2, s=max(order2.shape[0] * 0.06, 1.0))
        xc2 = np.logspace(np.log10(x2.min() + 1), np.log10(x2.max() + 1), 120)
        ax.plot(xc2, spl2(np.log10(xc2 + 1)), color="0.12", lw=2.0, zorder=3)
    except Exception as e:
        print("Fig3D spline failed:", e)
    ax.set_xscale("log")
    ax.set_xlabel(r"Training presence-only sample size $n_{\mathrm{PO}}$ (log scale)", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\Delta$AUC (B-spline IPP $-$ maxnet)", fontsize=FS_LABEL)
    ax.set_title(
        "D  ΔAUC (B-spline IPP − maxnet) versus sample size\n"
        "(points = species; black line = spline smoother)",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FS_TICK)
    handles = [Line2D([0], [0], marker="o", color=REGION_COLOR[r], markersize=7, lw=0, label=r) for r in REGION_ORDER]
    # lower-right: cloud is denser mid-left / upper; clears the smoother end
    ax.legend(
        handles=handles, fontsize=FS_LEG, loc="lower right", ncol=2,
        title="Region", title_fontsize=FS_LEG, frameon=False,
    )
    _despine(ax)

    fig.tight_layout(pad=0.6, h_pad=1.2, w_pad=1.0)
    _save(fig, OUT_MS, "Fig3_additive_robustness")


def plot_fig4_curve_panels() -> None:
    """Write Fig.4 *panels* for manual collage. Does not touch Fig4_response_curves.png."""
    agr_path = CURVE_DIR / "nceas_kan_vs_gam_agreement_all.csv"
    if not agr_path.is_file():
        print("WARN: curve agreement CSV missing; skip Fig4 panels:", agr_path)
        return
    agr = pd.read_csv(agr_path)
    bad = agr.nsmallest(3, "pearson_r")
    good = agr[agr.pearson_r > 0.9999].sample(3, random_state=1)
    rv = np.sort(agr.pearson_r.values)

    # A: ECDF
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    yecdf = np.arange(1, len(rv) + 1) / len(rv)
    ax.step(rv, yecdf, where="post", color=COLOR_ADDITIVE, lw=1.6)
    ax.fill_between(rv, yecdf, step="post", color=COLOR_ADDITIVE, alpha=0.12)
    for p, yp in [(50, 0.50), (90, 0.90), (95, 0.95)]:
        t = float(np.percentile(rv, p))
        ax.axhline(yp, color=COLOR_MAXNET, lw=0.55, ls="--", alpha=0.45)
        ax.text(0.46, yp + 0.018, f"P{p}={t:.4f}", fontsize=8.5, color=COLOR_MAXNET)
    o = agr[agr.pearson_r < 0.5]
    if len(o):
        ax.annotate(
            f"swi05 sfroyy r={o.pearson_r.values[0]:.3f}",
            xy=(o.pearson_r.values[0], 1.0 / len(agr)),
            xytext=(0.58, 0.18),
            arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
            fontsize=8.5,
            color="0.3",
        )
    ax.set_xlabel("Pearson r (B-spline IPP vs GAM-IPP)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("A  Response-curve agreement (n = 159)")
    ax.set_xlim(0.44, 1.005)
    ax.set_ylim(0, 1.05)
    _despine(ax)
    _save(fig, OUT_MS, "Fig4_A_ecdf")

    # B: regional means — regional pastels; axis starts at 0.90 (not 0.92)
    fig, ax = plt.subplots(figsize=(3.7, 3.3))
    rr = agr.groupby("region").pearson_r.mean().reindex(REGION_ORDER)
    xb = np.arange(len(REGION_ORDER))
    cols = [REGION_COLOR[r] for r in REGION_ORDER]
    ax.bar(xb, rr.values, color=cols, edgecolor="0.25", lw=0.6, alpha=0.92)
    for xi, mu_val in zip(xb, rr.values):
        ax.text(xi, float(mu_val) + 0.0025, f"{float(mu_val):.3f}", ha="center", fontsize=9, color="0.2")
    ax.set_xticks(xb)
    ax.set_xticklabels(REGION_ORDER)
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("B  By region  (axis from 0.90)")
    ax.set_ylim(0.90, 1.008)
    _despine(ax)
    _save(fig, OUT_MS, "Fig4_B_bars")

    # C1–C6: exemplar curves (method colours)
    labels_c = ["Divergent"] * 3 + ["Typical"] * 3
    rows_c = [bad.iloc[i] for i in range(3)] + [good.iloc[i] for i in range(3)]
    for i, (label, prow) in enumerate(zip(labels_c, rows_c)):
        fig, ax = plt.subplots(figsize=(3.3, 2.55))
        reg, sp, feat = prow["region"], prow["species_id"], prow["feature"]
        r_val = prow["pearson_r"]
        kp = CURVE_DIR / "nceas" / reg / sp / "kan_ipp" / f"phi_{feat}.csv"
        gp = CURVE_DIR / "nceas" / reg / sp / "gam_ipp" / f"phi_{feat}.csv"
        if kp.exists() and gp.exists():
            kan = pd.read_csv(kp)
            gam = pd.read_csv(gp)
            msk = kan["in_train_support"] == 1
            ax.plot(
                kan.loc[msk, "x_raw"],
                kan.loc[msk, "phi"],
                color=COLOR_ADDITIVE,
                lw=1.6,
                zorder=3,
                label="B-spline IPP",
            )
            ax.plot(
                gam.loc[msk, "x_raw"],
                gam.loc[msk, "phi"],
                color=COLOR_GAM,
                lw=1.6,
                ls="--",
                zorder=3,
                label="GAM-IPP",
            )
        ax.set_title(f"{label}, r={r_val:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{reg} / {sp} / {feat}", fontsize=10, style="italic")
        ax.tick_params(labelsize=8.5)
        _despine(ax)
        ax.legend(fontsize=8, loc="best", frameon=False)
        _save(fig, OUT_MS, f"Fig4_C{i + 1}")


def plot_fig4_curves_locked() -> None:
    """§1.4 response curves: manual collage — never regenerate or overwrite.

    Canonical file: figures/ms_results/v7/Fig4_response_curves.png
    Optional backup name: Fig6.png (same bytes historically).
    """
    OUT_MS.mkdir(parents=True, exist_ok=True)
    locked = OUT_MS / "Fig4_response_curves.png"
    backup = OUT_MS / "Fig6.png"
    if locked.is_file():
        print("Fig.4 LOCKED (manual collage present):", locked)
        return
    if backup.is_file():
        import shutil
        shutil.copy2(backup, locked)
        print("Fig.4 restored from Fig6.png backup (still manual; not regenerated):", locked)
        return
    print("WARN: Fig.4 manual collage missing — place Fig4_response_curves.png under", OUT_MS)


def plot_fig5_deep_with_e2e(df: pd.DataFrame, wide: pd.DataFrame) -> None:
    """§1.5 residual: (A) Deep-2 vs Add, (B) Deep-3 vs Add, (C) Deep-2 Rφ vs e2e.

    三栏并排时每个子图物理宽度小，故提高字号并加高画布（阅读反馈：此前极小）。
    """
    add = wide.set_index(["region", "species_id"])["additive_kan_ipp"]
    e2e = e2e_itt(df).set_index(["region", "species_id"])
    e2e_auc = e2e["auc_roc"]

    # 1×3 layout fixed; larger type for on-screen readability
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEG = 14.0, 13.0, 12.0, 11.0

    arms_d2 = [
        ("B", "deep2_rphi", "Deep-2 Rφ", COLOR_DEEP2_RPHI),
        ("B", "deep2_rx", "Deep-2 Rx", COLOR_DEEP2_RX),
    ]
    arms_d3 = [
        ("C", "deep3_rphi", "Deep-3 Rφ", COLOR_DEEP3_RPHI),
        ("C", "deep3_rx", "Deep-3 Rx", COLOR_DEEP3_RX),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), sharey=False)

    def _region_box(ax, stage_models, title, baseline: pd.Series):
        positions, data, colors, tick_pos, tick_lab = [], [], [], [], []
        pos = 1.0
        for reg in REGION_ORDER:
            for stage, model, lab, col in stage_models:
                s = deep_species_mean(df, stage, model)
                common = s.index.intersection(baseline.index)
                d = (s.loc[common] - baseline.loc[common]).reset_index()
                d.columns = ["region", "species_id", "delta"]
                dd = d.loc[d.region == reg, "delta"].values
                dd = dd[np.isfinite(dd)]
                data.append(dd)
                colors.append(col)
                positions.append(pos)
                pos += 0.9
            tick_pos.append(pos - 1.35)
            tick_lab.append(reg)
            pos += 0.55
        bp = ax.boxplot(
            data, positions=positions, widths=0.7, patch_artist=True,
            showfliers=False, manage_ticks=False,
        )
        for i, (patch, c) in enumerate(zip(bp["boxes"], colors)):
            patch.set_facecolor(c)
            patch.set_alpha(0.90)
            patch.set_edgecolor("0.15")
            # Even index = Rφ (solid); odd = Rx (hatched)
            if i % 2 == 1:
                patch.set_hatch("///")
                patch.set_linewidth(0.9)
        ax.axhline(0, color="0.35", ls="--", lw=0.8)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=FS_TICK)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.set_title(title, fontsize=FS_TITLE)
        _annotate_region_pair_medians(
            ax,
            data,
            tick_pos=tick_pos,
            face_colors=colors,
            fontsize=7.5,
            fmt="{:+.3f}",
        )
        legend_handles = []
        for j, (stage, model, lab, col) in enumerate(stage_models):
            s = deep_species_mean(df, stage, model)
            common = s.index.intersection(baseline.index)
            mu, lo, hi = _boot_mean_ci((s.loc[common] - baseline.loc[common]).values)
            legend_handles.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=col,
                    edgecolor="0.15",
                    hatch="///" if j == 1 else None,
                    label=f"{lab}: mean Δ = {mu:+.3f}",
                )
            )
        ax.legend(handles=legend_handles, frameon=False, fontsize=FS_LEG, loc="lower left")
        _despine(ax)

    _region_box(
        axes[0],
        arms_d2,
        "A  Fair residual Deep-2 − Additive baseline\n"
        "ΔAUC by region (Rφ vs Rx mixer inputs; 3 seeds)",
        add,
    )
    axes[0].set_ylabel("ΔAUC (Deep − Additive)", fontsize=FS_LABEL)
    _region_box(
        axes[1],
        arms_d3,
        "B  Fair residual Deep-3 − Additive baseline\n"
        "ΔAUC by region (Rφ vs Rx mixer inputs; 3 seeds)",
        add,
    )
    axes[1].set_ylabel("ΔAUC (Deep − Additive)", fontsize=FS_LABEL)

    # C: Deep-2 Rφ − e2e by region
    ax = axes[2]
    s = deep_species_mean(df, "B", "deep2_rphi")
    common = s.index.intersection(e2e_auc.index)
    d = (s.loc[common] - e2e_auc.loc[common]).reset_index()
    d.columns = ["region", "species_id", "delta"]
    deltas, labels, cols = [], [], []
    for reg in REGION_ORDER:
        dd = d.loc[d.region == reg, "delta"].values
        dd = dd[np.isfinite(dd)]
        if len(dd) == 0:
            continue
        deltas.append(dd)
        labels.append(reg)
        cols.append(REGION_COLOR[reg])
    bp = ax.boxplot(deltas, patch_artist=True, widths=0.55, showfliers=False)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)
    ax.axhline(0, color="0.35", ls="--", lw=0.8)
    mu, lo, hi = _boot_mean_ci(d["delta"].values)
    ax.set_title(
        "C  Deep-2 Rφ − standard KAN end-to-end (ITT)\n"
        f"ΔAUC by region (mean = {mu:+.3f}; additive backbone retained)",
        fontsize=FS_TITLE,
    )
    ax.set_ylabel("ΔAUC (Deep-2 Rφ − e2e)", fontsize=FS_LABEL)
    _annotate_boxplot_medians(
        ax, deltas, fontsize=7.0, fmt="{:+.3f}", face_colors=cols
    )
    _despine(ax)

    fig.tight_layout(pad=0.5, w_pad=1.2)
    _save(fig, OUT_MS, "Fig5_deepkan_ablation")


def archive_legacy_stems() -> None:
    """Move old numbered intermediates out of the way of main-text Fig.1–5."""
    import shutil

    arch = OUT_MS / "_archive_legacy"
    arch.mkdir(parents=True, exist_ok=True)
    legacy_stems = [
        "Fig2_species_AUC_boxplots",
        "Fig3_paired_comparisons",
        "Fig4_species_ranking",
        "Fig5_nPO_effect",
        "Fig6",
        "Fig6_A_ecdf",
        "Fig6_B_bars",
        "Fig6_C1",
        "Fig6_C2",
        "Fig6_C3",
        "Fig6_C4",
        "Fig6_C5",
        "Fig6_C6",
    ]
    for stem in legacy_stems:
        for ext in ("png", "pdf"):
            p = OUT_MS / f"{stem}.{ext}"
            if p.is_file():
                dest = arch / p.name
                # do not overwrite a newer archive copy with empty; always refresh
                shutil.move(str(p), str(dest))
                print("archived", p.name, "->", dest)


def main() -> None:
    _setup_style()
    df = load_metrics()
    wide = stage_a_wide(df)
    print("stage A species", len(wide))

    # Main-text figures only (canonical stems)
    plot_fig1_e2e_maxnet(df)
    plot_fig2_performance_composite(wide, df)
    plot_fig3_robustness_composite(wide)
    plot_fig4_curve_panels()
    plot_fig4_curves_locked()
    plot_fig5_deep_with_e2e(df, wide)

    # SI + tables
    plot_si_figures(df, wide)
    write_si_tables(df, wide)
    write_si_md()

    # Legacy intermediates (optional rebuild into archive for debug)
    plot_fig2_auc_box(wide)
    plot_fig3_paired(wide, df)
    plot_fig4_ranking(wide)
    plot_fig5_npo(wide)
    # Move non-canonical numbered files away from main-text namespace
    # Keep Fig4_response_curves and Fig1/2/3/5 composites in place.
    archive_legacy_stems()

    print("done.")


if __name__ == "__main__":
    main()
