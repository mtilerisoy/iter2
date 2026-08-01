"""Figures for the shuffle/noise control in ``results/control_clap_*.csv``.

The question the control answers: when removing a head raises AUROC, is that because the
head's *specific arrangement* of weights was doing something, or merely because the
network was perturbed (a regularisation-flavoured, capacity-style change)?

Three conditions are compared against the intact model, per (dataset, head):

    removed    the head's contribution zeroed
    shuffled   the same weights, permuted within the head — arrangement destroyed,
               magnitudes and count preserved
    noise      Gaussian noise at the parameters' own scale — arrangement preserved,
               magnitudes disturbed

If gains were generic regularisation, ``noise`` would reproduce them. If arrangement is
what matters, ``shuffled`` tracks ``removed`` and ``noise`` does not.

Usage:
    python analyze_control.py
    python analyze_control.py --formats png pdf
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_prune import (
    AXIS,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    Line2D,
    plt,
    save,
    style_axes,
    titles,
)

# Three conditions = three categorical slots, the most that validate on every pair.
CONDITION_COLOR = {"removed": "#2a78d6", "shuffled": "#eb6834", "noise": "#1baf7a"}
CONDITION_ORDER = ["removed", "shuffled", "noise"]
SELECTION_TITLE = {
    "helpful": "Heads whose removal HELPED",
    "harmful": "Heads whose removal HURT",
}


def load_control(path):
    """Long frame of effects, with the per-dataset intact AUROC subtracted."""
    data = pd.read_csv(path)
    intact = (
        data[data["condition"] == "intact"].set_index("dataset")["AUROC"].to_dict()
    )
    trials = data[data["condition"] != "intact"].copy()
    trials["intact"] = trials["dataset"].map(intact)
    trials["effect"] = trials["AUROC"] - trials["intact"]
    return trials, intact


def head_order(subset):
    """Helpful heads first (best on the left), then harmful (worst on the right)."""
    ranking = (
        subset.groupby(["selection", "pruning_id"])["sweep_effect"].first().reset_index()
    )
    helpful = ranking[ranking["selection"] == "helpful"].sort_values("sweep_effect", ascending=False)
    harmful = ranking[ranking["selection"] == "harmful"].sort_values("sweep_effect")
    return helpful["pruning_id"].tolist() + harmful["pruning_id"].tolist()


def fig_by_head(trials, out_dir, formats):
    """One panel per dataset: every head, every condition, against the intact line."""
    datasets = sorted(trials["dataset"].unique())
    cols = min(3, len(datasets))
    rows = int(np.ceil(len(datasets) / cols))
    limit = float(np.nanmax(np.abs(trials["effect"]))) * 1.15

    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.1 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, dataset in zip(axes, datasets):
        subset = trials[trials["dataset"] == dataset]
        heads = head_order(subset)
        positions = {head: index for index, head in enumerate(heads)}

        # Shade the harmful half so the two-sided reading is immediate.
        n_helpful = subset[subset["selection"] == "helpful"]["pruning_id"].nunique()
        if 0 < n_helpful < len(heads):
            ax.axvspan(n_helpful - 0.5, len(heads) - 0.5, color=GRID, alpha=0.4, zorder=0)

        for condition, offset in zip(CONDITION_ORDER, (-0.26, 0.0, 0.26)):
            rows_c = subset[subset["condition"] == condition]
            if rows_c.empty:
                continue
            x = np.array([positions[head] for head in rows_c["pruning_id"]]) + offset
            if condition == "removed":
                ax.scatter(
                    x, rows_c["effect"].to_numpy(), s=52, marker="D",
                    color=CONDITION_COLOR[condition], zorder=5, linewidths=0,
                )
                continue
            # Distribution of the 10 draws: every point, plus its median as a wide tick.
            jitter = np.random.default_rng(0).uniform(-0.07, 0.07, len(x))
            ax.scatter(
                x + jitter, rows_c["effect"].to_numpy(), s=13,
                color=CONDITION_COLOR[condition], alpha=0.55, linewidths=0, zorder=4,
            )
            medians = rows_c.groupby("pruning_id")["effect"].median()
            for head, value in medians.items():
                ax.plot(
                    [positions[head] + offset - 0.15, positions[head] + offset + 0.15],
                    [value, value], color=CONDITION_COLOR[condition], linewidth=2.2, zorder=6,
                )

        ax.axhline(0, color=AXIS, linewidth=1.2)
        ax.set_xticks(range(len(heads)), heads, rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(heads) - 0.4)
        ax.set_ylim(-limit, limit)
        ax.set_title(dataset, loc="left", fontsize=10)
        ax.text(
            1.0, 1.02, f"intact {subset['intact'].iloc[0]:.3f}",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK_MUTED,
        )
        style_axes(ax, grid_axis="y")

    for ax in axes[len(datasets):]:
        ax.set_visible(False)
    for index in range(0, len(datasets), cols):
        axes[index].set_ylabel("ΔAUROC vs intact")

    handles = [
        Line2D([], [], marker="D", linestyle="none", color=CONDITION_COLOR["removed"],
               markersize=7, label="removed (1 run)"),
        Line2D([], [], marker="o", linestyle="none", color=CONDITION_COLOR["shuffled"],
               markersize=6, label="shuffled (10 draws)"),
        Line2D([], [], marker="o", linestyle="none", color=CONDITION_COLOR["noise"],
               markersize=6, label="noise (10 draws)"),
        Line2D([], [], color=GRID, linewidth=8, label="shaded = removal hurt this head"),
    ]
    fig.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.005, -0.01), ncol=4,
        fontsize=9, labelcolor=INK_SECONDARY, frameon=False,
    )
    fig.suptitle(
        "Removing a head vs. only scrambling it vs. only perturbing it",
        x=0.005, y=1.0, ha="left", va="top", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.005, 0.978,
        "CLAP HTSAT · per dataset, the 3 heads whose removal helped most and the 3 whose removal hurt most",
        ha="left", va="top", fontsize=8.5, color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    save(fig, out_dir, "01_control_by_head", formats)


def fig_summary(trials, out_dir, formats):
    """The headline: mean effect per condition, split by whether removal helped or hurt."""
    per_head = (
        trials.groupby(["dataset", "pruning_id", "selection", "condition"])["effect"]
        .mean()
        .reset_index()
    )

    selections = [s for s in ("helpful", "harmful") if s in set(per_head["selection"])]
    fig, axes = plt.subplots(1, len(selections), figsize=(5.4 * len(selections), 4.9), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, selection in zip(axes, selections):
        subset = per_head[per_head["selection"] == selection]
        conditions = [c for c in CONDITION_ORDER if c in set(subset["condition"])]
        for position, condition in enumerate(conditions):
            values = subset[subset["condition"] == condition]["effect"].to_numpy()
            ax.bar(
                position, float(np.mean(values)), width=0.5,
                color=CONDITION_COLOR[condition], zorder=2,
            )
            jitter = np.random.default_rng(1).uniform(-0.13, 0.13, len(values))
            ax.scatter(
                position + jitter, values, s=16, color=INK_MUTED,
                alpha=0.65, linewidths=0, zorder=4,
            )
            ax.annotate(
                f"{np.mean(values):+.4f}", (position, np.mean(values)),
                textcoords="offset points", xytext=(0, 6 if np.mean(values) >= 0 else -14),
                ha="center", fontsize=9, color=INK_SECONDARY,
            )

        ax.axhline(0, color=AXIS, linewidth=1.2)
        ax.set_xticks(range(len(conditions)), conditions)
        ax.set_xlim(-0.6, len(conditions) - 0.4)
        ax.set_title(SELECTION_TITLE.get(selection, selection), loc="left", fontsize=10)
        style_axes(ax, grid_axis="y")

    axes[0].set_ylabel("ΔAUROC vs intact")
    fig.legend(
        handles=[Line2D([], [], marker="o", linestyle="none", color=INK_MUTED,
                        markersize=6, label="one (dataset, head)")],
        loc="lower left", bbox_to_anchor=(0.005, -0.01), fontsize=9,
        labelcolor=INK_SECONDARY, frameon=False,
    )
    fig.suptitle(
        "Does the gain survive when the head is scrambled or merely perturbed?",
        x=0.005, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK,
    )
    fig.text(
        0.005, 0.942,
        "Mean over datasets x heads; each control condition averaged over its 10 draws first",
        ha="left", va="top", fontsize=8.5, color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    save(fig, out_dir, "02_condition_summary", formats)


def fig_tracking(trials, out_dir, formats):
    """Does each control track removal head-by-head? Points on y=x mean 'same effect'."""
    per_head = (
        trials.groupby(["dataset", "pruning_id", "selection", "condition"])["effect"]
        .mean()
        .unstack("condition")
        .reset_index()
    )
    if "removed" not in per_head:
        return

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    span = float(np.nanmax(np.abs(per_head[[c for c in CONDITION_ORDER if c in per_head]].to_numpy()))) * 1.12
    ax.plot([-span, span], [-span, span], color=AXIS, linewidth=1.0, zorder=1)
    ax.annotate(
        "same effect as removing", (span * 0.62, span * 0.62), textcoords="offset points",
        xytext=(6, -12), fontsize=8.5, color=INK_MUTED,
    )
    ax.axhline(0, color=GRID, linewidth=1.0, zorder=0)
    ax.axvline(0, color=GRID, linewidth=1.0, zorder=0)

    for condition in ("shuffled", "noise"):
        if condition not in per_head:
            continue
        x = per_head["removed"].to_numpy()
        y = per_head[condition].to_numpy()
        # How much of removal's effect the control reproduces: correlation, and the
        # slope through the origin (1.0 would mean "exactly as strong as removal").
        r = float(np.corrcoef(x, y)[0, 1])
        slope = float(np.dot(x, y) / np.dot(x, x))
        ax.scatter(
            x, y, s=42, color=CONDITION_COLOR[condition], alpha=0.75, linewidths=0,
            zorder=3, label=f"{condition}  (r={r:+.2f}, slope={slope:.2f})",
        )
        print(f"  {condition:<9} vs removed: r={r:+.3f}  slope={slope:.3f}")

    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_xlabel("ΔAUROC from removing the head")
    ax.set_ylabel("ΔAUROC from the control (mean of 10 draws)")
    titles(
        ax, "Does the control reproduce removal?",
        "One point per (dataset, head) · on the diagonal = indistinguishable from removal",
    )
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK_SECONDARY)
    style_axes(ax, grid_axis="both")
    save(fig, out_dir, "03_control_vs_removal", formats)


def write_summary(trials, out_dir):
    """The numbers behind the figures, so a claim can be checked without re-plotting."""
    per_head = (
        trials.groupby(["dataset", "pruning_id", "selection", "condition"])["effect"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    per_head.to_csv(out_dir / "control_per_head.csv", index=False, float_format="%.6f")

    summary = (
        per_head.pivot_table(index="selection", columns="condition", values="mean", aggfunc="mean")
        .reindex(columns=[c for c in CONDITION_ORDER if c in set(per_head["condition"])])
    )
    summary.to_csv(out_dir / "control_summary.csv", float_format="%.6f")
    print("\nMean ΔAUROC vs intact:")
    print(summary.round(4).to_string())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--formats", nargs="+", default=["png"])
    args = parser.parse_args()

    paths = sorted(Path(args.results_dir).glob("control_clap_*.csv"))
    if not paths:
        raise SystemExit(f"No control_clap_*.csv in {args.results_dir}/")

    for path in paths:
        trials, _ = load_control(path)
        pooling = trials["pooling"].iloc[0]
        seed = trials["seed"].iloc[0]
        out_dir = Path(args.figures_dir) / f"control_{pooling}_seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{path.name} -> figures/{out_dir.name}/")
        print(
            f"  {trials['dataset'].nunique()} datasets x "
            f"{trials.groupby('dataset')['pruning_id'].nunique().max()} heads, "
            f"{len(trials)} runs"
        )

        fig_by_head(trials, out_dir, args.formats)
        fig_summary(trials, out_dir, args.formats)
        fig_tracking(trials, out_dir, args.formats)
        write_summary(trials, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
