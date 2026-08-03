"""Is "this head helps this dataset" stable across splits, or an artifact of one split?

The sweep was run twice on the same recordings with the two available splits in opposite
roles:

    forward   k-NN fitted on train, scored on test   (the standard protocol)
    reverse   k-NN fitted on test,  scored on train

Both directions reuse the same embeddings; only which split fits the classifier and which
is scored changes. Crucially the two AUROCs are measured on **disjoint recordings**, so
correlating the 184-head effect vectors within a dataset asks exactly one thing: is the
per-head effect a property of the (head, dataset) pair, or of the particular recordings
that happened to be scored?

    rho >> 0   the head x dataset interaction is real and repeatable
    rho ~ 0    the effect is split-specific movement, and the per-head claims do not hold

The extreme heads get their own test, because a claim only ever rested on them: take the
heads the forward split calls most helpful, and see what the reverse split says about
those same heads. Selection uses forward only, so the reverse numbers are honest.

Usage:
    python analyze_split_stability.py --formats png pdf
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
    SERIES,
    SERIES_SOFT,
    Line2D,
    plt,
    save,
    style_axes,
    titles,
)

HELPFUL_COLOR = "#2a78d6"
HARMFUL_COLOR = "#eb6834"


def effects(results_dir, pooling, seed, suffix=""):
    sweep = pd.read_csv(Path(results_dir) / f"prune_clap_head_{pooling}_seed{seed}{suffix}.csv")
    baseline = pd.read_csv(Path(results_dir) / f"prune_clap_none_{pooling}_seed{seed}{suffix}.csv")
    sweep["effect"] = sweep["AUROC"] - sweep["dataset"].map(baseline.set_index("dataset")["AUROC"])
    return sweep.pivot_table(index="pruning_id", columns="dataset", values="effect").sort_index()


def stability(forward, reverse, top_k):
    """Per dataset: agreement between the two splits, over all heads and over the extremes."""
    records = []
    for dataset in forward.columns:
        if dataset not in reverse.columns:
            continue
        a = forward[dataset]
        b = reverse[dataset].reindex(a.index)
        joint = pd.concat([a.rename("forward"), b.rename("reverse")], axis=1).dropna()

        rho, p_rho = stats.spearmanr(joint["forward"], joint["reverse"])
        r, p_r = stats.pearsonr(joint["forward"], joint["reverse"])

        # Selection on forward only; the reverse column is an out-of-sample readout.
        helps = joint.nlargest(top_k, "forward")
        hurts = joint.nsmallest(top_k, "forward")
        extremes = pd.concat([helps, hurts])
        rho_extreme, _ = stats.spearmanr(extremes["forward"], extremes["reverse"])

        records.append(
            {
                "dataset": dataset,
                "spearman": float(rho),
                "spearman_p": float(p_rho),
                "pearson": float(r),
                "pearson_p": float(p_r),
                "spearman_extremes": float(rho_extreme),
                "forward_helps_mean": float(helps["forward"].mean()),
                "reverse_of_forward_helps": float(helps["reverse"].mean()),
                "forward_hurts_mean": float(hurts["forward"].mean()),
                "reverse_of_forward_hurts": float(hurts["reverse"].mean()),
                "reverse_all_mean": float(joint["reverse"].mean()),
                "sign_agreement": float(np.mean(np.sign(joint["forward"]) == np.sign(joint["reverse"]))),
                "sign_agreement_extremes": float(
                    np.mean(np.sign(extremes["forward"]) == np.sign(extremes["reverse"]))
                ),
                "n_heads": int(len(joint)),
            }
        )
    return pd.DataFrame(records)


def survival(forward, reverse, top_k, n_permutations=10000, seed=0):
    """How much of the forward effect is still there on the other split?

    A correlation is attenuated by noise in either direction, which makes a low value hard
    to interpret. This asks the question in AUROC units instead: take the heads the
    forward split singles out, read their mean effect on the reverse split, and compare it
    against the mean of random head sets of the same size drawn from the same reverse
    column. That floor is the honest null — it is what any k heads do to this dataset.
    """
    rng = np.random.default_rng(seed)
    records = []
    for dataset in forward.columns:
        if dataset not in reverse.columns:
            continue
        a = forward[dataset]
        b = reverse[dataset].reindex(a.index)
        joint = pd.concat([a.rename("forward"), b.rename("reverse")], axis=1).dropna()
        values = joint["reverse"].to_numpy()

        for side, selected in (
            ("helps", joint.nlargest(top_k, "forward")),
            ("hurts", joint.nsmallest(top_k, "forward")),
        ):
            observed = float(selected["reverse"].mean())
            draws = np.array([
                values[rng.choice(len(values), size=top_k, replace=False)].mean()
                for _ in range(n_permutations)
            ])
            centre = float(draws.mean())
            # One-sided in the direction the forward split predicts.
            if side == "helps":
                p_value = float((draws >= observed).sum() + 1) / (n_permutations + 1)
            else:
                p_value = float((draws <= observed).sum() + 1) / (n_permutations + 1)
            forward_mean = float(selected["forward"].mean())
            records.append(
                {
                    "dataset": dataset,
                    "side": side,
                    "forward_mean": forward_mean,
                    "reverse_mean": observed,
                    "random_floor": centre,
                    "reverse_above_floor": observed - centre,
                    "survival_fraction": (observed - centre) / forward_mean if forward_mean else np.nan,
                    "p_permutation": p_value,
                }
            )
    return pd.DataFrame(records)


def fig_survival(surv, out_dir, formats, top_k):
    """Forward effect, reverse effect, and the random floor — in AUROC units."""
    datasets = sorted(surv["dataset"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)

    for ax, side, colour in zip(axes, ("helps", "hurts"), (HELPFUL_COLOR, HARMFUL_COLOR)):
        subset = surv[surv["side"] == side].set_index("dataset").reindex(datasets)
        ypos = np.arange(len(datasets))
        ax.barh(ypos - 0.19, subset["forward_mean"], height=0.34, color=colour, zorder=2,
                label="forward split (the selection)")
        ax.barh(ypos + 0.19, subset["reverse_mean"], height=0.34, color=SERIES_SOFT, zorder=2,
                label="reverse split (out of sample)")
        ax.scatter(subset["random_floor"], ypos + 0.19, s=34, marker="|", color=INK,
                   zorder=4, label="random-head floor, reverse split")
        for y, p_value in zip(ypos, subset["p_permutation"]):
            if p_value < 0.05:
                ax.annotate("*", (subset["reverse_mean"].iloc[y], y + 0.19),
                            textcoords="offset points", xytext=(9, -4), fontsize=13, color=INK)
        ax.axvline(0, color=AXIS, linewidth=1.0)
        ax.set_yticks(ypos, datasets)
        ax.set_xlabel("mean ΔAUROC")
        ax.set_title(f"forward top-{top_k} '{side}' heads", loc="left", fontsize=10)
        style_axes(ax, grid_axis="x")
    axes[0].legend(fontsize=8.5, labelcolor=INK_SECONDARY, loc="lower right")

    fig.suptitle("How much of the effect survives the split swap?",
                 x=0.005, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.005, 0.945,
             "* = the reverse effect beats the random-head floor (permutation p < 0.05)",
             ha="left", va="top", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, out_dir, "04_effect_survival", formats)


def fig_scatter(forward, reverse, table, out_dir, formats, top_k):
    """One panel per dataset: the same head's effect under each split."""
    datasets = list(table["dataset"])
    cols = min(4, len(datasets))
    rows = int(np.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.7 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, dataset in zip(axes, datasets):
        a = forward[dataset]
        b = reverse[dataset].reindex(a.index)
        row = table[table["dataset"] == dataset].iloc[0]

        helps = set(a.nlargest(top_k).index)
        hurts = set(a.nsmallest(top_k).index)
        colours = [
            HELPFUL_COLOR if head in helps else HARMFUL_COLOR if head in hurts else SERIES_SOFT
            for head in a.index
        ]
        sizes = [30 if head in helps or head in hurts else 12 for head in a.index]
        ax.scatter(a, b, s=sizes, c=colours, alpha=0.75, linewidths=0, zorder=3)

        span = float(np.nanmax(np.abs(np.concatenate([a.to_numpy(), b.to_numpy()])))) * 1.1
        ax.plot([-span, span], [-span, span], color=AXIS, linewidth=1.0, zorder=1)
        ax.axhline(0, color=GRID, linewidth=1.0, zorder=0)
        ax.axvline(0, color=GRID, linewidth=1.0, zorder=0)
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_xlabel("ΔAUROC, forward split")
        ax.set_ylabel("ΔAUROC, reverse split")
        ax.set_title(f"{dataset}   ρ={row['spearman']:+.2f}", loc="left", fontsize=10)
        style_axes(ax, grid_axis="both")

    for ax in axes[len(datasets):]:
        ax.set_visible(False)

    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=HELPFUL_COLOR, markersize=7,
                   label=f"forward top-{top_k} helps"),
            Line2D([], [], marker="o", linestyle="none", color=HARMFUL_COLOR, markersize=7,
                   label=f"forward top-{top_k} hurts"),
            Line2D([], [], marker="o", linestyle="none", color=SERIES_SOFT, markersize=6,
                   label="all other heads"),
            Line2D([], [], color=AXIS, linewidth=1.4, label="perfect agreement"),
        ],
        loc="lower left", bbox_to_anchor=(0.005, -0.01), ncol=4, fontsize=9,
        labelcolor=INK_SECONDARY, frameon=False,
    )
    fig.suptitle("Does the same head do the same thing under the other split?",
                 x=0.005, y=1.0, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.005, 0.975,
             "One point per head · axes are measured on disjoint recordings",
             ha="left", va="top", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.955))
    save(fig, out_dir, "01_split_scatter", formats)


def fig_rho_bars(table, out_dir, formats, top_k):
    """The headline number per dataset, all heads and extremes only."""
    ordered = table.sort_values("spearman")
    ypos = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(ordered) + 2.4))

    ax.barh(ypos - 0.19, ordered["spearman"], height=0.34, color=SERIES, zorder=2,
            label="all 184 heads")
    ax.barh(ypos + 0.19, ordered["spearman_extremes"], height=0.34, color=SERIES_SOFT, zorder=2,
            label=f"forward extremes only (top/bottom {top_k})")

    # Where a correlation stops being distinguishable from zero at n=184.
    threshold = 1.96 / np.sqrt(184 - 3)
    ax.axvspan(-threshold, threshold, color=GRID, alpha=0.7, zorder=0)
    ax.axvline(0, color=AXIS, linewidth=1.0)
    ax.set_yticks(ypos, ordered["dataset"])
    ax.set_xlabel("Spearman ρ between the two splits' head-effect vectors")
    titles(
        ax, "Split-stability of the per-head effect",
        f"grey band = |ρ| not distinguishable from zero at n=184 (±{threshold:.2f})",
    )
    ax.legend(fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")
    style_axes(ax, grid_axis="x")
    save(fig, out_dir, "02_split_rho", formats)


def fig_extreme_readout(table, out_dir, formats, top_k):
    """What the reverse split says about the heads the forward split singled out."""
    datasets = list(table["dataset"])
    fig, ax = plt.subplots(figsize=(max(8.5, 1.3 * len(datasets) + 3), 4.8))

    for position, dataset in enumerate(datasets):
        row = table[table["dataset"] == dataset].iloc[0]
        ax.scatter(position - 0.16, row["forward_helps_mean"], s=64, marker="D",
                   color=HELPFUL_COLOR, zorder=4)
        ax.scatter(position + 0.16, row["reverse_of_forward_helps"], s=64, marker="o",
                   color=HELPFUL_COLOR, alpha=0.55, zorder=4)
        ax.scatter(position - 0.16, row["forward_hurts_mean"], s=64, marker="D",
                   color=HARMFUL_COLOR, zorder=4)
        ax.scatter(position + 0.16, row["reverse_of_forward_hurts"], s=64, marker="o",
                   color=HARMFUL_COLOR, alpha=0.55, zorder=4)
        for value_a, value_b, colour in (
            (row["forward_helps_mean"], row["reverse_of_forward_helps"], HELPFUL_COLOR),
            (row["forward_hurts_mean"], row["reverse_of_forward_hurts"], HARMFUL_COLOR),
        ):
            ax.plot([position - 0.16, position + 0.16], [value_a, value_b],
                    color=colour, linewidth=1.2, alpha=0.5, zorder=3)

    ax.axhline(0, color=AXIS, linewidth=1.2)
    ax.set_xticks(range(len(datasets)), datasets, fontsize=9)
    ax.set_ylabel("mean ΔAUROC")
    titles(
        ax, f"The forward split's top-{top_k} heads, re-read on the other split",
        "diamond = forward (the selection, biased) · circle = reverse (out of sample)",
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="D", linestyle="none", color=HELPFUL_COLOR, markersize=8,
                   label="selected as helps — forward"),
            Line2D([], [], marker="o", linestyle="none", color=HELPFUL_COLOR, markersize=8,
                   alpha=0.55, label="same heads — reverse"),
            Line2D([], [], marker="D", linestyle="none", color=HARMFUL_COLOR, markersize=8,
                   label="selected as hurts — forward"),
            Line2D([], [], marker="o", linestyle="none", color=HARMFUL_COLOR, markersize=8,
                   alpha=0.55, label="same heads — reverse"),
        ],
        loc="upper left", bbox_to_anchor=(0, -0.12), ncol=4, fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    style_axes(ax, grid_axis="y")
    save(fig, out_dir, "03_extreme_readout", formats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--pooling", default="projected")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--formats", nargs="+", default=["png"])
    args = parser.parse_args()

    forward = effects(args.results_dir, args.pooling, args.seed)
    reverse = effects(args.results_dir, args.pooling, args.seed, suffix="_reverse")
    print(f"forward {forward.shape}, reverse {reverse.shape}")

    table = stability(forward, reverse, args.top_k)
    out_dir = Path(args.figures_dir) / f"split_stability_{args.pooling}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "split_stability.csv", index=False, float_format="%.6f")

    print("\nSplit stability per dataset:")
    print(table[["dataset", "spearman", "spearman_p", "spearman_extremes",
                 "sign_agreement", "sign_agreement_extremes"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nmean ρ over datasets: {table['spearman'].mean():+.3f} "
          f"(median {table['spearman'].median():+.3f})")
    threshold = 1.96 / np.sqrt(184 - 3)
    print(f"|ρ| distinguishable from zero at n=184: > {threshold:.3f}")
    print(f"datasets above that: {int((table['spearman'].abs() > threshold).sum())}/{len(table)}")

    print(f"\nThe forward split's top-{args.top_k} 'helps' heads, read on the reverse split:")
    print(table[["dataset", "forward_helps_mean", "reverse_of_forward_helps",
                 "forward_hurts_mean", "reverse_of_forward_hurts", "reverse_all_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    surv = survival(forward, reverse, args.top_k)
    surv.to_csv(out_dir / "effect_survival.csv", index=False, float_format="%.6f")
    print(f"\nSurvival of the forward top-{args.top_k}, against a random-head floor on the reverse split:")
    print(surv.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    fig_scatter(forward, reverse, table, out_dir, args.formats, args.top_k)
    fig_survival(surv, out_dir, args.formats, args.top_k)
    fig_rho_bars(table, out_dir, args.formats, args.top_k)
    fig_extreme_readout(table, out_dir, args.formats, args.top_k)
    print("\nDone.")


if __name__ == "__main__":
    main()
