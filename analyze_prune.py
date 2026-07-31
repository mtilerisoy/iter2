"""Figures for the structured-pruning sweeps in ``results/prune_clap_*.csv``.

Every sweep CSV is discovered automatically, matched against the ``--prune_type none``
reference of the same (pooling, seed), and turned into a folder of figures under
``figures/``. Head sweeps and block sweeps are handled by the same code path; dense
sweeps (184 heads) get line/heatmap treatments, small ones (12 blocks) get labelled bars.

Everything is plotted as an **effect**: ``AUROC(pruned) - AUROC(intact)`` for the same
dataset. Negative means the ablated unit was carrying signal. The scale is diverging with
a neutral midpoint because the quantity has a meaningful zero and a meaningful sign.

Usage:
    python analyze_prune.py                       # every sweep in results/
    python analyze_prune.py --results-dir results --figures-dir figures
    python analyze_prune.py --formats png pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- palette
# Single documented palette, used by role. Only one categorical hue is ever on screen
# at once (identity is carried by axis labels and panel titles, never by hue), so no
# multi-series palette needs validating here.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"  # categorical slot 1 (blue)
SERIES_SOFT = "#9ec5f4"
NEGATIVE = "#e34948"  # diverging red pole: ablation hurts
POSITIVE = "#2a78d6"  # diverging blue pole: ablation helps
MIDPOINT = "#f0efec"

# Diverging ramp, equal steps per arm, neutral gray midpoint (never a hue at the middle).
DIVERGING = LinearSegmentedColormap.from_list(
    "prune_effect",
    ["#8f2020", "#c23434", "#e34948", "#f3a3a2", MIDPOINT, "#9ec5f4", "#3987e5", "#2a78d6", "#184f95"],
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "legend.frameon": False,
        "figure.dpi": 110,
    }
)


def style_axes(ax, grid_axis="y"):
    """Hairline, recessive chrome: no box, grid on one axis only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid_axis in ("x", "both"):
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    if grid_axis in ("y", "both"):
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)


def titles(ax, title, subtitle):
    """Title then subtitle, both offset in points so they never collide at any axes height."""
    ax.set_title(title, loc="left", pad=24)
    ax.annotate(
        subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7),
        textcoords="offset points", ha="left", va="bottom",
        fontsize=8.5, color=INK_MUTED, annotation_clip=False,
    )


def thinned(candidates, min_gap):
    """Keep the strongest labels only, dropping any that would sit on top of a kept one."""
    kept = []
    for _, row in candidates.reindex(candidates["mean"].abs().sort_values(ascending=False).index).iterrows():
        if all(abs(row["pruning_index"] - other["pruning_index"]) >= min_gap for other in kept):
            kept.append(row)
    return kept


def save(fig, out_dir, stem, formats):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {out_dir.name}/{stem}.{'/'.join(formats)}")


# ------------------------------------------------------------------------------ data


def load_results(results_dir):
    """All sweep rows in one frame, plus the intact-reference AUROC per (dataset, pooling, seed)."""
    frames = [pd.read_csv(path) for path in sorted(Path(results_dir).glob("prune_clap_*.csv"))]
    if not frames:
        raise SystemExit(f"No prune_clap_*.csv found in {results_dir}/")
    data = pd.concat(frames, ignore_index=True)

    reference = data[data["pruning_type"] == "none"]
    sweeps = data[data["pruning_type"] != "none"]
    baseline = reference.set_index(["dataset", "pooling", "seed"])["AUROC"]
    return sweeps, baseline, reference


def attach_effect(sweep, baseline):
    """effect = AUROC(pruned) - AUROC(intact). Rows without a reference row are dropped."""
    sweep = sweep.copy()
    keys = list(zip(sweep["dataset"], sweep["pooling"], sweep["seed"]))
    sweep["baseline"] = [baseline.get(key, np.nan) for key in keys]
    missing = sorted(sweep.loc[sweep["baseline"].isna(), "dataset"].unique())
    if missing:
        print(f"  warning: no --prune_type none reference for {missing}; dropped")
    sweep = sweep.dropna(subset=["baseline"])
    sweep["effect"] = sweep["AUROC"] - sweep["baseline"]
    return sweep


def unit_table(sweep):
    """Per-unit effect aggregated across datasets, in sweep order."""
    table = (
        sweep.groupby(["pruning_index", "pruning_id"])["effect"]
        .agg(mean="mean", std="std", worst="min", best="max", n="size")
        .reset_index()
        .sort_values("pruning_index")
    )
    table["stage"] = table["pruning_id"].str.extract(r"s(\d+)").astype(int)
    table["block"] = table["pruning_id"].str.extract(r"s\d+\.b(\d+)").astype(int)
    return table


def stage_spans(table):
    """(stage, first_index, last_index) for shading and separators."""
    return [
        (int(stage), int(group["pruning_index"].min()), int(group["pruning_index"].max()))
        for stage, group in table.groupby("stage")
    ]


def symmetric_limit(values):
    """Symmetric colour/axis limit so the neutral midpoint sits exactly at zero."""
    return float(np.nanmax(np.abs(values))) or 1e-6


# --------------------------------------------------------------------------- figures


def fig_baseline(reference, pooling, seed, out_dir, formats):
    ref = reference[(reference["pooling"] == pooling) & (reference["seed"] == seed)]
    ref = ref.sort_values("AUROC", ascending=True)
    if ref.empty:
        return

    fig, ax = plt.subplots(figsize=(6.8, 0.38 * len(ref) + 1.7))
    ypos = np.arange(len(ref))
    ax.barh(ypos, ref["AUROC"].to_numpy(), height=0.5, color=SERIES)
    ax.set_yticks(ypos, ref["dataset"].tolist())
    ax.axvline(0.5, color=AXIS, linewidth=1.0)
    ax.text(0.5, -0.75, " chance", color=INK_MUTED, fontsize=8, va="center")

    for y, value in zip(ypos, ref["AUROC"].to_numpy()):
        ax.text(value + 0.008, y, f"{value:.3f}", va="center", fontsize=8.5, color=INK_SECONDARY)

    ax.set_xlim(0, 1.06)
    ax.set_xlabel("k-NN AUROC (k=5, cosine)")
    titles(ax, "Intact CLAP audio tower",
           f"{pooling} embeddings · seed {seed} · reference for every ablation")
    style_axes(ax, grid_axis="x")
    save(fig, out_dir, "01_baseline_auroc", formats)


def fig_heatmap(sweep, table, meta, out_dir, formats):
    datasets = sorted(sweep["dataset"].unique())
    matrix = (
        sweep.pivot_table(index="dataset", columns="pruning_index", values="effect")
        .reindex(index=datasets, columns=table["pruning_index"])
        .to_numpy()
    )
    # A couple of extreme cells would otherwise flatten every other cell to near-white.
    limit = float(np.nanpercentile(np.abs(matrix), 98)) or symmetric_limit(matrix)
    clipped = float(np.nanmax(np.abs(matrix))) > limit
    dense = len(table) > 20

    fig, ax = plt.subplots(figsize=(13.5 if dense else 8.0, 0.42 * len(datasets) + 2.4))
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
    )

    ax.set_yticks(range(len(datasets)), datasets)
    if dense:
        spans = stage_spans(table)
        positions = {index: i for i, index in enumerate(table["pruning_index"])}
        for _, _, last in spans[:-1]:
            ax.axvline(positions[last] + 0.5, color=SURFACE, linewidth=2.0)
        ax.set_xticks(
            [(positions[first] + positions[last]) / 2 for _, first, last in spans],
            [f"stage {stage}" for stage, _, _ in spans],
        )
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_xticks(range(len(table)), table["pruning_id"].tolist(), rotation=45, ha="right", fontsize=8)

    ax.set_xlabel(f"ablated {meta['unit']}" + (" (in sweep order)" if dense else ""))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.03, extend="both" if clipped else "neither")
    bar.set_label("ΔAUROC vs intact model", color=INK_SECONDARY, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(color=AXIS, labelcolor=INK_SECONDARY)

    titles(
        ax, f"Effect of removing one {meta['unit']} at a time",
        f"{meta['subtitle']} · red = removing it hurts · blue = removing it helps"
        + (f" · colour scale clipped at ±{limit:.3f}" if clipped else ""),
    )
    save(fig, out_dir, "02_effect_heatmap", formats)


def fig_mean_effect(sweep, table, meta, out_dir, formats):
    dense = len(table) > 20
    fig, ax = plt.subplots(figsize=(13.5 if dense else 8.5, 4.6))

    # Every dataset's own effect, deliberately in one muted ink: the spread is the
    # message, not which dataset is which (that is figure 05).
    ax.scatter(
        sweep["pruning_index"].to_numpy(), sweep["effect"].to_numpy(),
        s=9, color=SERIES_SOFT, alpha=0.55, linewidths=0, zorder=2,
    )

    if dense:
        for stage, first, last in stage_spans(table)[1::2]:
            ax.axvspan(first - 0.5, last + 0.5, color=GRID, alpha=0.35, zorder=0)
        for stage, first, last in stage_spans(table):
            ax.text(
                (first + last) / 2, 0.985, f"stage {stage}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.5, color=INK_MUTED,
            )

    # Descriptive spread of all single-unit effects. Not a significance test — there is
    # one measurement per (dataset, unit), so nothing here supports a confidence claim.
    effects = sweep["effect"].to_numpy()
    mad = float(np.median(np.abs(effects - np.median(effects))))
    ax.axhspan(-mad, mad, color=GRID, alpha=0.55, zorder=1)
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=3)

    if dense:
        ax.plot(table["pruning_index"].to_numpy(), table["mean"].to_numpy(), color=SERIES, linewidth=1.8, zorder=4)
    else:
        colors = [NEGATIVE if value < 0 else POSITIVE for value in table["mean"]]
        ax.bar(table["pruning_index"].to_numpy(), table["mean"].to_numpy(), width=0.62, color=colors, zorder=4)
        ax.set_xticks(table["pruning_index"].to_numpy(), table["pruning_id"].tolist(), rotation=45, ha="right", fontsize=8)

    # Direct-label only the extremes, and only where the labels do not overlap.
    candidates = pd.concat([table.nsmallest(4, "mean"), table.nlargest(3, "mean")])
    for row in thinned(candidates, min_gap=len(table) * 0.05):
        ax.annotate(
            row["pruning_id"],
            (row["pruning_index"], row["mean"]),
            textcoords="offset points", xytext=(0, -15 if row["mean"] < 0 else 10),
            ha="center", fontsize=8, color=INK_SECONDARY,
        )

    ax.set_xlabel(f"ablated {meta['unit']}" + (" (sweep index)" if dense else ""))
    ax.set_ylabel("ΔAUROC vs intact model")
    titles(ax, f"Mean effect per {meta['unit']}, with every dataset behind it", meta["subtitle"])
    ax.legend(
        handles=[
            Line2D([], [], color=SERIES, linewidth=1.8, label=f"mean over {sweep['dataset'].nunique()} datasets")
            if dense
            else Line2D([], [], color=SERIES, linewidth=6, label=f"mean over {sweep['dataset'].nunique()} datasets"),
            Line2D([], [], marker="o", linestyle="none", color=SERIES_SOFT, markersize=5, label="one dataset"),
            Line2D([], [], color=GRID, linewidth=8, label=f"±1 MAD of all effects ({mad:.3f})"),
        ],
        loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, fontsize=8.5, labelcolor=INK_SECONDARY,
    )
    style_axes(ax, grid_axis="y")
    save(fig, out_dir, "03_mean_effect", formats)


def fig_by_block(sweep, table, meta, out_dir, formats):
    """Where in the network sensitivity lives, aggregated per transformer block."""
    merged = sweep.merge(table[["pruning_index", "stage", "block"]], on="pruning_index")
    merged["label"] = "s" + merged["stage"].astype(str) + ".b" + merged["block"].astype(str)
    labels = (
        merged[["stage", "block", "label"]]
        .drop_duplicates()
        .sort_values(["stage", "block"])["label"]
        .tolist()
    )
    groups = [merged.loc[merged["label"] == label, "effect"].to_numpy() for label in labels]

    fig, ax = plt.subplots(figsize=(max(7.0, 0.72 * len(labels) + 2.0), 4.4))
    parts = ax.boxplot(
        groups, positions=np.arange(len(labels)), widths=0.55, patch_artist=True,
        showfliers=False, medianprops=dict(color=INK, linewidth=1.4),
        whiskerprops=dict(color=AXIS, linewidth=1.0), capprops=dict(color=AXIS, linewidth=1.0),
    )
    for patch in parts["boxes"]:
        patch.set_facecolor(SERIES_SOFT)
        patch.set_edgecolor(AXIS)
        patch.set_linewidth(0.8)

    rng = np.random.default_rng(0)
    for position, values in enumerate(groups):
        ax.scatter(
            position + rng.uniform(-0.16, 0.16, len(values)), values,
            s=7, color=SERIES, alpha=0.35, linewidths=0, zorder=3,
        )

    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=8.5)
    ax.set_ylabel("ΔAUROC vs intact model")
    ax.set_xlabel("transformer block (stage.block)")
    titles(ax, f"Effect distribution by block — every {meta['unit']} × dataset", meta["subtitle"])
    style_axes(ax, grid_axis="y")
    save(fig, out_dir, "04_effect_by_block", formats)


def fig_per_dataset(sweep, table, meta, out_dir, formats):
    datasets = sorted(sweep["dataset"].unique())
    dense = len(table) > 20
    cols = 2 if len(datasets) <= 4 else 3
    rows = int(np.ceil(len(datasets) / cols))
    limit = symmetric_limit(sweep["effect"]) * 1.08

    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 2.5 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, dataset in zip(axes, datasets):
        subset = sweep[sweep["dataset"] == dataset].sort_values("pruning_index")
        if dense:
            for _, first, last in stage_spans(table)[1::2]:
                ax.axvspan(first - 0.5, last + 0.5, color=GRID, alpha=0.35, zorder=0)
        if dense:
            ax.plot(subset["pruning_index"].to_numpy(), subset["effect"].to_numpy(), color=SERIES, linewidth=1.0)
        else:
            colors = [NEGATIVE if value < 0 else POSITIVE for value in subset["effect"]]
            ax.bar(subset["pruning_index"].to_numpy(), subset["effect"].to_numpy(), width=0.62, color=colors)
        ax.axhline(0, color=AXIS, linewidth=0.9)
        ax.set_ylim(-limit, limit)
        ax.set_title(
            f"{dataset}   ", loc="left", fontsize=10,
        )
        ax.text(
            1.0, 1.02, f"intact {subset['baseline'].iloc[0]:.3f}",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK_MUTED,
        )
        style_axes(ax, grid_axis="y")

    for ax in axes[len(datasets):]:
        ax.set_visible(False)
    # With shared x, only the last *visible* panel in each column carries tick labels —
    # so the x label goes there too, never on a panel whose ticks are hidden.
    for column in range(cols):
        last = max((i for i in range(len(datasets)) if i % cols == column), default=None)
        if last is not None:
            axes[last].tick_params(labelbottom=True)
            axes[last].set_xlabel(f"ablated {meta['unit']} (sweep index)")
    for index in range(0, len(datasets), cols):
        axes[index].set_ylabel("ΔAUROC")

    fig.suptitle(
        f"Per-dataset ablation profile — one {meta['unit']} removed at a time",
        x=0.005, y=1.0, ha="left", va="top", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(0.005, 0.972, meta["subtitle"], ha="left", va="top", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, out_dir, "05_per_dataset_profiles", formats)


def fig_extremes(sweep, table, meta, out_dir, formats, top=8):
    top = min(top, len(table) // 2)
    if top < 1:
        return
    picked = pd.concat([table.nsmallest(top, "mean"), table.nlargest(top, "mean")])
    picked = picked.sort_values("mean", ascending=False)  # most damaging ends up on top

    fig, ax = plt.subplots(figsize=(8.6, 0.28 * len(picked) + 1.9))
    ypos = np.arange(len(picked))
    ax.barh(
        ypos, picked["mean"].to_numpy(), height=0.6,
        color=[NEGATIVE if value < 0 else POSITIVE for value in picked["mean"]],
    )

    lookup = sweep.set_index("pruning_index")
    for y, index in zip(ypos, picked["pruning_index"]):
        values = np.atleast_1d(lookup.loc[index, "effect"])
        ax.scatter(values, np.full(len(values), y), s=9, color=INK_MUTED, alpha=0.6, linewidths=0, zorder=3)

    ax.axvline(0, color=AXIS, linewidth=1.0)
    ax.set_yticks(ypos, picked["pruning_id"].tolist(), fontsize=8.5)
    ax.set_xlabel("ΔAUROC vs intact model (mean over datasets)")
    titles(
        ax, f"Most and least damaging {meta['unit']}s",
        f"{meta['subtitle']} · dots are individual datasets — a bar without agreement is noise",
    )
    ax.legend(
        handles=[
            Line2D([], [], color=NEGATIVE, linewidth=6, label="removing it hurts"),
            Line2D([], [], color=POSITIVE, linewidth=6, label="removing it helps"),
            Line2D([], [], marker="o", linestyle="none", color=INK_MUTED, markersize=5, label="one dataset"),
        ],
        loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, fontsize=8.5, labelcolor=INK_SECONDARY,
    )
    style_axes(ax, grid_axis="x")
    save(fig, out_dir, "06_extreme_units", formats)


# ------------------------------------------------------------------------------- run


def analyse_sweep(sweep, reference, meta, out_dir, formats):
    table = unit_table(sweep)
    print(f"\n{meta['title']} -> figures/{out_dir.name}/")
    print(f"  {len(table)} {meta['unit']}s x {sweep['dataset'].nunique()} datasets = {len(sweep)} rows")

    fig_baseline(reference, meta["pooling"], meta["seed"], out_dir, formats)
    fig_heatmap(sweep, table, meta, out_dir, formats)
    fig_mean_effect(sweep, table, meta, out_dir, formats)
    fig_by_block(sweep, table, meta, out_dir, formats)
    fig_per_dataset(sweep, table, meta, out_dir, formats)
    fig_extremes(sweep, table, meta, out_dir, formats)

    ranked = table.sort_values("mean")[
        ["pruning_index", "pruning_id", "stage", "block", "mean", "std", "worst", "best", "n"]
    ]
    ranked.to_csv(out_dir / "unit_effects.csv", index=False, float_format="%.6f")
    print(f"  {out_dir.name}/unit_effects.csv")


FIGURE_INDEX = [
    ("01_baseline_auroc", "Intact-model AUROC per dataset — the reference every other figure subtracts."),
    ("02_effect_heatmap", "Dataset x unit grid of ΔAUROC. The whole sweep on one screen."),
    ("03_mean_effect", "Mean effect per unit in sweep order, with every dataset's own value behind it."),
    ("04_effect_by_block", "Effect distribution per transformer block — where sensitivity concentrates."),
    ("05_per_dataset_profiles", "One panel per dataset; shows whether datasets agree at all."),
    ("06_extreme_units", "The most and least damaging units, with per-dataset dots as a consistency check."),
    ("unit_effects.csv", "Per-unit mean/std/worst/best effect across datasets, ranked."),
]


def write_index(figures_dir, produced):
    lines = [
        "# Figures",
        "",
        "Generated by `analyze_prune.py` from `results/prune_clap_*.csv` — do not edit by hand:",
        "",
        "```bash",
        "sbatch scripts/analyze-prune.sh --formats png pdf",
        "```",
        "",
        "Every figure plots an **effect**: `AUROC(pruned) - AUROC(intact)` for the same dataset,",
        "seed and pooling. Negative (red) means the ablated unit was carrying signal.",
        "",
    ]
    for name, count, datasets in produced:
        lines += [f"## `{name}/`", "", f"{count} ablated units x {datasets} datasets.", ""]
        lines += [f"- `{stem}` — {what}" for stem, what in FIGURE_INDEX]
        lines += [""]
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    (Path(figures_dir) / "README.md").write_text("\n".join(lines))
    print(f"\nWrote {figures_dir}/README.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--formats", nargs="+", default=["png"], help="png, pdf, svg ...")
    args = parser.parse_args()

    sweeps, baseline, reference = load_results(args.results_dir)
    if sweeps.empty:
        raise SystemExit("Only --prune_type none results found; nothing to sweep over.")

    produced = []
    for (prune_type, pooling, seed), group in sweeps.groupby(["pruning_type", "pooling", "seed"]):
        group = attach_effect(group, baseline)
        if group.empty:
            continue
        meta = {
            "unit": "head" if prune_type == "head" else prune_type,
            "pooling": pooling,
            "seed": seed,
            "title": f"{prune_type} sweep ({pooling}, seed {seed})",
            "subtitle": f"CLAP HTSAT audio tower · {pooling} embeddings · k-NN AUROC · seed {seed}",
        }
        out_dir = Path(args.figures_dir) / f"{prune_type}_{pooling}_seed{seed}"
        analyse_sweep(group, reference, meta, out_dir, args.formats)
        produced.append((out_dir.name, group["pruning_index"].nunique(), group["dataset"].nunique()))

    write_index(args.figures_dir, produced)
    print("\nDone.")


if __name__ == "__main__":
    main()
