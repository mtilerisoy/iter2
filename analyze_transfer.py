"""Is "which heads are harmful" shared across tasks, or task-specific — and how is it organised?

Pure re-analysis of the head sweep: no forward passes. Each dataset gives a vector of 184
head effects (ΔAUROC from removing that head). Correlating those vectors across datasets
answers the question the earlier experiments kept tripping over — the same head helps one
task and hurts another — by turning it into the measurement.

    high correlation everywhere   ->  head harmfulness is a property of the model
    block structure               ->  it is a property of the task family
    uniform near-zero             ->  it is per-(head, dataset) noise, and the whole
                                      line of work is weaker than it looks

The families are anatomical, fixed before looking at the matrix: the cardiac manifests are
auscultation recordings (BMD's Mit/Pul/Aor/Tri sites, CirCor's MV/AV/PV/TV, CinC's
PhysioNet heart sounds, ZCHSound's congenital heart disease), the respiratory ones are
lung-sound recordings. The clustering is run independently of that labelling so the two
can be compared rather than assumed.

Usage:
    python analyze_transfer.py
    python analyze_transfer.py --formats png pdf
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from analyze_prune import (
    AXIS,
    DIVERGING,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    SERIES_SOFT,
    SURFACE,
    Line2D,
    plt,
    save,
    style_axes,
    titles,
)

# Fixed before inspecting the matrix; see the module docstring for the evidence.
FAMILIES = {
    "BMD": "cardiac",
    "CinC": "cardiac",
    "CirCor": "cardiac",
    "ZCHSound": "cardiac",
    "ICBHI": "respiratory",
    "KAUH": "respiratory",
    "SPRSound": "respiratory",
}
FAMILY_COLOR = {"cardiac": "#2a78d6", "respiratory": "#eb6834"}


def effect_matrix(results_dir, pooling, seed):
    """[184 heads x n datasets] of ΔAUROC, plus the head order."""
    sweep = pd.read_csv(Path(results_dir) / f"prune_clap_head_{pooling}_seed{seed}.csv")
    baseline = pd.read_csv(Path(results_dir) / f"prune_clap_none_{pooling}_seed{seed}.csv")
    sweep["effect"] = sweep["AUROC"] - sweep["dataset"].map(baseline.set_index("dataset")["AUROC"])
    matrix = sweep.pivot_table(index="pruning_id", columns="dataset", values="effect")
    return matrix.sort_index()


def correlation_matrix(matrix, method="spearman"):
    """Dataset x dataset rank correlation of the head-effect vectors, with p-values."""
    datasets = list(matrix.columns)
    rho = pd.DataFrame(np.eye(len(datasets)), index=datasets, columns=datasets)
    pval = pd.DataFrame(np.zeros((len(datasets),) * 2), index=datasets, columns=datasets)
    for a, b in itertools.combinations(datasets, 2):
        if method == "spearman":
            r, p = stats.spearmanr(matrix[a], matrix[b], nan_policy="omit")
        else:
            r, p = stats.pearsonr(matrix[a], matrix[b])
        rho.loc[a, b] = rho.loc[b, a] = float(r)
        pval.loc[a, b] = pval.loc[b, a] = float(p)
    return rho, pval


def family_contrast(rho):
    """Within-family vs cross-family correlation, with an exact label-permutation test.

    With seven datasets every relabelling can be enumerated, so the p-value is exact
    rather than sampled: how often does a random split into groups of the same sizes
    separate within from across at least as well as the anatomical one?
    """
    datasets = list(rho.index)
    pairs = list(itertools.combinations(datasets, 2))
    values = np.array([rho.loc[a, b] for a, b in pairs])
    families = np.array([FAMILIES.get(d) for d in datasets])

    def statistic(labels):
        same = np.array([labels[datasets.index(a)] == labels[datasets.index(b)] for a, b in pairs])
        if same.all() or not same.any():
            return np.nan
        return float(values[same].mean() - values[~same].mean())

    observed = statistic(families)
    sizes = pd.Series(families).value_counts()
    null = []
    for chosen in itertools.combinations(range(len(datasets)), int(sizes.iloc[0])):
        labels = np.array(["b"] * len(datasets), dtype=object)
        labels[list(chosen)] = "a"
        value = statistic(labels)
        if not np.isnan(value):
            null.append(value)
    null = np.array(null)
    p_value = float((np.abs(null) >= abs(observed)).sum()) / len(null)

    same_mask = np.array([FAMILIES.get(a) == FAMILIES.get(b) for a, b in pairs])
    return {
        "observed_gap": observed,
        "within_mean": float(values[same_mask].mean()),
        "across_mean": float(values[~same_mask].mean()),
        "p_exact": p_value,
        "n_partitions": len(null),
        "pairs": pd.DataFrame(
            {
                "a": [a for a, _ in pairs],
                "b": [b for _, b in pairs],
                "rho": values,
                "same_family": same_mask,
            }
        ),
    }


def cluster_order(rho):
    """Average-linkage clustering on 1 - rho; returns the leaf order and the linkage."""
    distance = 1.0 - rho.to_numpy()
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2
    linkage = hierarchy.linkage(squareform(distance, checks=False), method="average")
    order = hierarchy.leaves_list(linkage)
    return [rho.index[i] for i in order], linkage


def transfer_matrix(matrix, k):
    """Entry (A, B): mean effect on B of the top-k heads that helped most on A.

    The full-vector correlation weights all 184 heads equally, and most of them do
    nothing — their effects sit at the noise floor, which attenuates any real structure.
    This statistic instead asks the directional question with the signal-carrying tail:
    take the heads A says are worth removing, and see what they do to B. Selection uses
    only A, so every off-diagonal entry is honest; the diagonal is selection-biased by
    construction and is reported only for scale.
    """
    datasets = list(matrix.columns)
    transfer = pd.DataFrame(index=datasets, columns=datasets, dtype=float)
    for source in datasets:
        chosen = matrix[source].nlargest(k).index
        for target in datasets:
            transfer.loc[source, target] = float(matrix.loc[chosen, target].mean())
    transfer.index.name = "selected_on"
    return transfer


def transfer_family_summary(transfer):
    """Off-diagonal transfer, split by whether source and target share a family."""
    records = []
    for source in transfer.index:
        for target in transfer.columns:
            if source == target:
                continue
            records.append(
                {
                    "selected_on": source,
                    "evaluated_on": target,
                    "mean_effect": float(transfer.loc[source, target]),
                    "same_family": FAMILIES.get(source) == FAMILIES.get(target),
                }
            )
    frame = pd.DataFrame(records)
    within = frame.loc[frame["same_family"], "mean_effect"]
    across = frame.loc[~frame["same_family"], "mean_effect"]
    statistic, p_value = stats.mannwhitneyu(within, across, alternative="two-sided")
    return frame, {
        "within_mean": float(within.mean()),
        "across_mean": float(across.mean()),
        "gap": float(within.mean() - across.mean()),
        "p_mannwhitney": float(p_value),
        "n_within": int(len(within)),
        "n_across": int(len(across)),
    }


# --------------------------------------------------------------------------- figures


def fig_matrix(rho, pval, order, out_dir, formats, n_heads):
    ordered = rho.loc[order, order]
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    limit = float(np.nanmax(np.abs(ordered.to_numpy() - np.eye(len(order)))))
    image = ax.imshow(ordered.to_numpy(), cmap=DIVERGING, vmin=-limit, vmax=limit)

    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color=INK_MUTED)
                continue
            value = ordered.iloc[i, j]
            marker = "*" if pval.loc[a, b] * (len(order) * (len(order) - 1) / 2) < 0.05 else ""
            ax.text(j, i, f"{value:+.2f}{marker}", ha="center", va="center", fontsize=9,
                    color=INK if abs(value) < limit * 0.6 else SURFACE)

    labels = [f"{name}" for name in order]
    ax.set_xticks(range(len(order)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(order)), labels)
    for tick, name in zip(ax.get_xticklabels(), order):
        tick.set_color(FAMILY_COLOR.get(FAMILIES.get(name), INK_SECONDARY))
    for tick, name in zip(ax.get_yticklabels(), order):
        tick.set_color(FAMILY_COLOR.get(FAMILIES.get(name), INK_SECONDARY))

    # Family blocks, drawn only where the clustered order happens to keep a family together.
    boundaries = [i for i in range(1, len(order))
                  if FAMILIES.get(order[i]) != FAMILIES.get(order[i - 1])]
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color=SURFACE, linewidth=2.5)
        ax.axvline(boundary - 0.5, color=SURFACE, linewidth=2.5)

    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    bar.set_label("Spearman ρ of head-effect vectors", color=INK_SECONDARY)
    bar.outline.set_visible(False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    titles(
        ax, "Does a head's effect on one task predict its effect on another?",
        f"{n_heads} heads ranked per dataset · blue = shared ranking · "
        f"* = significant after Bonferroni · label colour = "
        f"cardiac / respiratory",
    )
    save(fig, out_dir, "01_cross_task_matrix", formats)


def fig_family_contrast(contrast, out_dir, formats):
    pairs = contrast["pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    groups = [
        ("within cardiac", pairs[(pairs["same_family"]) & (pairs["a"].map(FAMILIES) == "cardiac")]),
        ("within respiratory", pairs[(pairs["same_family"]) & (pairs["a"].map(FAMILIES) == "respiratory")]),
        ("across families", pairs[~pairs["same_family"]]),
    ]
    for position, (name, subset) in enumerate(groups):
        colour = FAMILY_COLOR.get(name.split()[-1], INK_MUTED)
        ax.bar(position, subset["rho"].mean(), width=0.5, color=colour, zorder=2)
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(subset))
        ax.scatter(position + jitter, subset["rho"], s=26, color=INK_MUTED, alpha=0.75,
                   linewidths=0, zorder=4)
        for x, (_, row) in zip(position + jitter, subset.iterrows()):
            ax.annotate(f"{row['a']}–{row['b']}", (x, row["rho"]), textcoords="offset points",
                        xytext=(7, -3), fontsize=7, color=INK_MUTED)
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xticks(range(len(groups)), [name for name, _ in groups])
    ax.set_ylabel("Spearman ρ between head-effect vectors")
    ax.set_title("Pairwise correlation by family", loc="left", fontsize=10)
    style_axes(ax, grid_axis="y")

    ax = axes[1]
    ax.set_title("Is the anatomical split special?", loc="left", fontsize=10)
    ax.text(
        0.0, 0.72,
        f"within-family mean ρ   {contrast['within_mean']:+.3f}\n"
        f"across-family mean ρ   {contrast['across_mean']:+.3f}\n"
        f"gap                    {contrast['observed_gap']:+.3f}\n\n"
        f"exact permutation p = {contrast['p_exact']:.3f}\n"
        f"over all {contrast['n_partitions']} relabellings of\n"
        f"the same group sizes",
        transform=ax.transAxes, fontsize=10.5, va="top", family="monospace", color=INK_SECONDARY,
    )
    ax.axis("off")

    fig.suptitle("Is head-harmfulness organised by acoustic task family?",
                 x=0.005, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out_dir, "02_family_contrast", formats)


def fig_pair_scatter(matrix, rho, out_dir, formats):
    """What the strongest and weakest correlations actually look like, head by head."""
    pairs = [(a, b, rho.loc[a, b]) for a, b in itertools.combinations(rho.index, 2)]
    strongest = max(pairs, key=lambda item: item[2])
    weakest = min(pairs, key=lambda item: abs(item[2]))
    most_negative = min(pairs, key=lambda item: item[2])

    shown = [strongest, weakest, most_negative]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, (a, b, value) in zip(axes, shown):
        ax.scatter(matrix[a], matrix[b], s=22, color=SERIES, alpha=0.55, linewidths=0)
        ax.axhline(0, color=AXIS, linewidth=0.9)
        ax.axvline(0, color=AXIS, linewidth=0.9)
        ax.set_xlabel(f"ΔAUROC on {a}")
        ax.set_ylabel(f"ΔAUROC on {b}")
        family = "same family" if FAMILIES.get(a) == FAMILIES.get(b) else "different families"
        ax.set_title(f"{a} vs {b}   ρ={value:+.2f}\n{family}", loc="left", fontsize=9.5)
        style_axes(ax, grid_axis="both")

    fig.suptitle("The strongest, the weakest and the most negative pair",
                 x=0.005, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.005, 0.94, "One point per head · each axis is that head's effect on one dataset",
             ha="left", va="top", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, out_dir, "03_pair_scatters", formats)


def fig_transfer(transfer, summary, out_dir, formats, k):
    """Directional transfer: heads chosen on one dataset, evaluated on every other."""
    datasets = list(transfer.index)
    values = transfer.to_numpy(dtype=float).copy()
    off_diagonal = values[~np.eye(len(datasets), dtype=bool)]
    limit = float(np.nanmax(np.abs(off_diagonal)))

    fig, ax = plt.subplots(figsize=(8.6, 6.9))
    masked = np.ma.masked_array(values, mask=np.eye(len(datasets), dtype=bool))
    image = ax.imshow(masked, cmap=DIVERGING, vmin=-limit, vmax=limit)

    for i, source in enumerate(datasets):
        for j, target in enumerate(datasets):
            value = values[i, j]
            if i == j:
                ax.text(j, i, f"({value:+.3f})", ha="center", va="center", fontsize=8,
                        color=INK_MUTED)
                continue
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8.5,
                    color=INK if abs(value) < limit * 0.6 else SURFACE)

    ax.set_xticks(range(len(datasets)), datasets, rotation=45, ha="right")
    ax.set_yticks(range(len(datasets)), datasets)
    for tick, name in zip(ax.get_xticklabels() + ax.get_yticklabels(), datasets * 2):
        tick.set_color(FAMILY_COLOR.get(FAMILIES.get(name), INK_SECONDARY))
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("heads selected on")

    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    bar.set_label("mean ΔAUROC of the transferred heads", color=INK_SECONDARY)
    bar.outline.set_visible(False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    titles(
        ax, f"Do the top-{k} most-helpful heads of one task help another?",
        f"diagonal parenthesised (selection-biased) · same family {summary['within_mean']:+.4f} "
        f"vs different {summary['across_mean']:+.4f}, p={summary['p_mannwhitney']:.2f}",
    )
    save(fig, out_dir, "05_transfer_matrix", formats)


def fig_dendrogram(rho, linkage, out_dir, formats):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    hierarchy.dendrogram(
        linkage, labels=list(rho.index), ax=ax, color_threshold=0,
        above_threshold_color=INK_MUTED,
    )
    for tick in ax.get_xticklabels():
        tick.set_color(FAMILY_COLOR.get(FAMILIES.get(tick.get_text()), INK_SECONDARY))
        tick.set_fontsize(9.5)
    ax.set_ylabel("1 − ρ (average linkage)")
    titles(ax, "Clustering the datasets by their head-effect profile",
           "Colour = anatomical family, which the clustering never saw")
    style_axes(ax, grid_axis="y")
    save(fig, out_dir, "04_dendrogram", formats)


# ------------------------------------------------------------------------------- run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--pooling", default="projected")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10,
                        help="Heads taken from the source dataset for the transfer matrix.")
    parser.add_argument("--formats", nargs="+", default=["png"])
    args = parser.parse_args()

    matrix = effect_matrix(args.results_dir, args.pooling, args.seed)
    print(f"{matrix.shape[0]} heads x {matrix.shape[1]} datasets")

    rho, pval = correlation_matrix(matrix)
    order, linkage = cluster_order(rho)
    contrast = family_contrast(rho)

    out_dir = Path(args.figures_dir) / f"transfer_{args.pooling}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_dir / "head_effect_matrix.csv", float_format="%.6f")
    rho.to_csv(out_dir / "cross_task_rho.csv", float_format="%.4f")
    pval.to_csv(out_dir / "cross_task_p.csv", float_format="%.4g")
    contrast["pairs"].to_csv(out_dir / "pairwise_rho.csv", index=False, float_format="%.4f")

    print("\nCross-task Spearman ρ:")
    print(rho.loc[order, order].round(3).to_string())
    print(f"\nclustered order: {order}")
    print(f"\nwithin-family mean ρ  {contrast['within_mean']:+.3f}")
    print(f"across-family mean ρ  {contrast['across_mean']:+.3f}")
    print(f"gap {contrast['observed_gap']:+.3f}, exact p = {contrast['p_exact']:.4f} "
          f"over {contrast['n_partitions']} relabellings")

    significant = [(a, b, rho.loc[a, b], pval.loc[a, b])
                   for a, b in itertools.combinations(rho.index, 2)
                   if pval.loc[a, b] * 21 < 0.05]
    print(f"\npairs significant after Bonferroni x21: {len(significant)}/21")
    for a, b, r, p in sorted(significant, key=lambda item: -item[2]):
        print(f"  {a:<9} {b:<9} rho={r:+.3f}  p={p:.2e}  "
              f"({'same' if FAMILIES.get(a) == FAMILIES.get(b) else 'cross'} family)")

    transfer = transfer_matrix(matrix, args.top_k)
    transfer_pairs, transfer_summary = transfer_family_summary(transfer)
    transfer.to_csv(out_dir / "transfer_matrix.csv", float_format="%.6f")
    transfer_pairs.to_csv(out_dir / "transfer_pairs.csv", index=False, float_format="%.6f")
    print(f"\nTransfer of the top-{args.top_k} helpful heads (rows: selected on, cols: evaluated on):")
    print(transfer.astype(float).round(4).to_string())
    print(f"\nsame-family transfer   {transfer_summary['within_mean']:+.4f} "
          f"(n={transfer_summary['n_within']})")
    print(f"cross-family transfer  {transfer_summary['across_mean']:+.4f} "
          f"(n={transfer_summary['n_across']})")
    print(f"gap {transfer_summary['gap']:+.4f}, Mann-Whitney p = {transfer_summary['p_mannwhitney']:.3f}")

    fig_matrix(rho, pval, order, out_dir, args.formats, matrix.shape[0])
    fig_transfer(transfer, transfer_summary, out_dir, args.formats, args.top_k)
    fig_family_contrast(contrast, out_dir, args.formats)
    fig_pair_scatter(matrix, rho, out_dir, args.formats)
    fig_dendrogram(rho, linkage, out_dir, args.formats)
    print("\nDone.")


if __name__ == "__main__":
    main()
