"""Figures for the held-out family-transfer test in ``results/family_transfer_*.csv``.

Every comparison is *within* a target dataset, because the transfer matrix showed a large
per-target main effect: some datasets gain from almost any perturbation and others lose.
The random condition is the control that makes that visible — it is what removing k
arbitrary heads does to this dataset, and no selected set is interesting unless it beats
that floor.

Usage:
    python analyze_family_transfer.py --formats png pdf
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze_prune import (
    AXIS,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    Line2D,
    plt,
    save,
    style_axes,
    titles,
)

CONDITION_COLOR = {
    "own_family": "#2a78d6",
    "other_family": "#eb6834",
    "universal": "#1baf7a",
}
CONDITION_LABEL = {
    "own_family": "selected on same family",
    "other_family": "selected on other family",
    "universal": "selected on all others",
}
ORDER = ["own_family", "other_family", "universal"]


def load(path):
    data = pd.read_csv(path)
    intact = data[data["condition"] == "intact"].set_index("dataset")["AUROC"]
    trials = data[data["condition"] != "intact"].copy()
    trials["intact"] = trials["dataset"].map(intact)
    trials["effect"] = trials["AUROC"] - trials["intact"]
    return trials


def summary_table(trials):
    table = (
        trials.groupby(["dataset", "family", "condition"])["effect"].mean().unstack()
    )
    return table[[c for c in ORDER + ["random"] if c in table.columns]]


def paired_tests(table):
    """Within-target paired comparisons — the only ones the design supports."""
    results = {}
    for left, right in (("own_family", "other_family"), ("own_family", "random"),
                        ("other_family", "random"), ("universal", "random")):
        if left not in table or right not in table:
            continue
        a, b = table[left].to_numpy(), table[right].to_numpy()
        results[f"{left} vs {right}"] = {
            "wins": int((a > b).sum()),
            "n": len(a),
            "mean_gap": float(np.mean(a - b)),
            "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue),
        }
    return results


def fig_per_target(trials, table, out_dir, formats, k):
    """Each target dataset: the three selected sets against the random-removal floor."""
    datasets = list(table.index.get_level_values("dataset"))
    fig, ax = plt.subplots(figsize=(max(9.0, 1.5 * len(datasets) + 3.0), 5.2))

    for position, dataset in enumerate(datasets):
        randoms = trials[(trials["dataset"] == dataset) & (trials["condition"] == "random")]["effect"]
        # The floor: what k arbitrary heads do to this dataset.
        ax.add_patch(
            plt.Rectangle(
                (position - 0.42, randoms.min()), 0.84, max(randoms.max() - randoms.min(), 1e-9),
                color=GRID, zorder=1,
            )
        )
        ax.hlines(randoms.mean(), position - 0.42, position + 0.42, color=INK_MUTED,
                  linewidth=1.4, zorder=2)
        ax.scatter(
            np.full(len(randoms), position) + np.random.default_rng(0).uniform(-0.3, 0.3, len(randoms)),
            randoms, s=12, color=INK_MUTED, alpha=0.5, linewidths=0, zorder=3,
        )

        for condition, offset in zip(ORDER, (-0.26, 0.0, 0.26)):
            row = table.xs(dataset, level="dataset")[condition]
            ax.scatter(position + offset, float(row.iloc[0]), s=70, marker="D",
                       color=CONDITION_COLOR[condition], zorder=5, linewidths=0)

    ax.axhline(0, color=AXIS, linewidth=1.2)
    families = [table.index.get_level_values("family")[i] for i in range(len(datasets))]
    ax.set_xticks(range(len(datasets)),
                  [f"{d}\n{f}" for d, f in zip(datasets, families)], fontsize=9)
    ax.set_ylabel("ΔAUROC vs intact model")
    titles(
        ax, f"Removing {k} heads chosen on other datasets — does the choice matter?",
        "Grey band and dots = 10 random head sets of the same size on the same dataset",
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="D", linestyle="none", color=CONDITION_COLOR[c],
                   markersize=8, label=CONDITION_LABEL[c]) for c in ORDER
        ] + [Line2D([], [], color=GRID, linewidth=10, label=f"{k} random heads (range of 10 draws)")],
        loc="upper left", bbox_to_anchor=(0, -0.12), ncol=4, fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    style_axes(ax, grid_axis="y")
    save(fig, out_dir, "01_transfer_per_target", formats)


def fig_paired(table, tests, out_dir, formats):
    """Own-family against other-family, target by target: the family hypothesis, paired."""
    datasets = list(table.index.get_level_values("dataset"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    for position, dataset in enumerate(datasets):
        own = float(table.xs(dataset, level="dataset")["own_family"].iloc[0])
        other = float(table.xs(dataset, level="dataset")["other_family"].iloc[0])
        ax.plot([0, 1], [own, other], color=GRID, linewidth=1.6, zorder=1)
        ax.scatter([0], [own], s=52, color=CONDITION_COLOR["own_family"], zorder=3, linewidths=0)
        ax.scatter([1], [other], s=52, color=CONDITION_COLOR["other_family"], zorder=3, linewidths=0)
        ax.annotate(dataset, (1, other), textcoords="offset points", xytext=(8, -3),
                    fontsize=8, color=INK_MUTED)
    ax.axhline(0, color=AXIS, linewidth=1.2)
    ax.set_xticks([0, 1], ["same family", "other family"])
    ax.set_xlim(-0.3, 1.6)
    ax.set_ylabel("ΔAUROC vs intact")
    ax.set_title("Each line is one target dataset", loc="left", fontsize=10)
    style_axes(ax, grid_axis="y")

    ax = axes[1]
    lines = [
        f"{name:<26} {stat['wins']}/{stat['n']}  gap {stat['mean_gap']:+.4f}  p={stat['wilcoxon_p']:.3f}"
        for name, stat in tests.items()
    ]
    ax.text(0.0, 0.85, "Within-target paired tests\n\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=9, va="top", family="monospace", color=INK_SECONDARY)
    ax.axis("off")

    fig.suptitle("Does selecting heads on the same acoustic family transfer better?",
                 x=0.005, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out_dir, "02_own_vs_other_family", formats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--pooling", default="projected")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--formats", nargs="+", default=["png"])
    args = parser.parse_args()

    path = Path(args.results_dir) / f"family_transfer_{args.pooling}_seed{args.seed}.csv"
    trials = load(path)
    table = summary_table(trials)
    tests = paired_tests(table)
    k = int(trials["k"].max())

    out_dir = Path(args.figures_dir) / f"family_transfer_{args.pooling}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "transfer_effects.csv", float_format="%.6f")

    print(f"ΔAUROC vs intact, removing {k} heads jointly (random = mean of 10 draws):")
    print(table.round(4).to_string())
    print("\nmean over targets:")
    print(table.mean().round(4).to_string())
    print("\nwithin-target paired tests:")
    for name, stat in tests.items():
        print(f"  {name:<28} {stat['wins']}/{stat['n']} wins, "
              f"mean gap {stat['mean_gap']:+.4f}, Wilcoxon p={stat['wilcoxon_p']:.3f}")
    pd.DataFrame(tests).T.to_csv(out_dir / "paired_tests.csv", float_format="%.4f")

    # How movable is each dataset by *any* perturbation? The control the earlier
    # experiments did not have.
    floor = trials[trials["condition"] == "random"].groupby("dataset")["effect"].agg(["mean", "std"])
    print(f"\nrandom {k}-head removal per dataset (the susceptibility floor):")
    print(floor.round(4).to_string())
    floor.to_csv(out_dir / "random_floor.csv", float_format="%.6f")

    fig_per_target(trials, table, out_dir, args.formats, k)
    fig_paired(table, tests, out_dir, args.formats)
    print("\nDone.")


if __name__ == "__main__":
    main()
