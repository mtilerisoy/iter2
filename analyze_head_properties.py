"""Which measurable head properties predict whether removing the head helps?

Joins the sweep effects with the head descriptors from ``eval_head_properties.py`` and
asks, for every property, whether it separates helpful-to-remove heads from harmful ones.

**The stage confound is the whole methodological problem here.** Head size, weight norm,
activation scale and pruning effect *all* vary systematically across the four HTSAT
stages, so a raw correlation between any property and the pruning effect can be nothing
more than "stage predicts both". Every statistic in this script is therefore reported
twice: raw across all 184 heads, and after z-scoring the property **within its stage**,
which removes the stage's mean and scale. Only the within-stage column is evidence about
heads; the raw column is kept so the size of the confound stays visible.

Statistics reported per property:

    rho        Spearman correlation with the pruning effect, over heads
    auc        probability a randomly chosen helpful head ranks above a harmful one
               (0.5 = no separation), computed from the top/bottom effect groups

Both are computed per dataset and then summarised, because "helpful to remove" is a
per-dataset property and the seven datasets do not agree with each other.

Usage:
    python analyze_head_properties.py
    python analyze_head_properties.py --formats png pdf --group-size 20
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

HELPFUL_COLOR = "#2a78d6"  # removal helps
HARMFUL_COLOR = "#eb6834"  # removal hurts

PROPERTY_LABELS = {
    "general_share": "general-audio energy share",
    "medical_share": "medical energy share",
    "generality": "generality (general − medical share)",
    "general_contrib_rms": "general-audio contribution RMS",
    "medical_contrib_rms": "medical contribution RMS",
    "contrib_ratio": "log2(general / medical contribution)",
    "w_norm": "weight norm",
    "w_norm_qkv": "QKV weight norm",
    "w_norm_out": "output-projection norm",
    "w_rms": "weight RMS",
    "w_kurtosis": "weight kurtosis",
    "w_participation": "weight participation ratio",
    "w_max_over_rms": "max |w| / RMS",
    "ov_spectral": "OV spectral norm",
}


def load(results_dir, pooling, seed):
    static = pd.read_csv(Path(results_dir) / f"head_static_{pooling}_seed{seed}.csv")
    activations = pd.read_csv(Path(results_dir) / f"head_activations_{pooling}_seed{seed}.csv")
    sweep = pd.read_csv(Path(results_dir) / f"prune_clap_head_{pooling}_seed{seed}.csv")
    baseline = pd.read_csv(Path(results_dir) / f"prune_clap_none_{pooling}_seed{seed}.csv")
    sweep["effect"] = sweep["AUROC"] - sweep["dataset"].map(baseline.set_index("dataset")["AUROC"])
    return static, activations, sweep


def head_features(static, activations):
    """One row per head: static descriptors plus medical/general activation summaries."""
    general = activations[activations["domain"] == "general"]
    medical = activations[activations["domain"] == "medical"]

    # Medical is averaged over the seven corpora so it is one comparable number per head;
    # a per-corpus version would just re-run the same analysis seven times.
    medical_mean = medical.groupby("pruning_id")[["contrib_rms", "ctx_rms", "energy_share"]].mean()
    general_mean = general.groupby("pruning_id")[["contrib_rms", "ctx_rms", "energy_share"]].mean()

    features = static.set_index("pruning_id").join(
        [
            medical_mean.add_prefix("medical_"),
            general_mean.add_prefix("general_"),
        ]
    )
    # Shares already normalise away each corpus's overall loudness, so their difference is
    # the cleanest "is this head relatively louder on general audio" contrast.
    features["generality"] = features["general_energy_share"] - features["medical_energy_share"]
    features["contrib_ratio"] = np.log2(
        features["general_contrib_rms"] / features["medical_contrib_rms"]
    )
    features = features.rename(
        columns={"medical_energy_share": "medical_share", "general_energy_share": "general_share"}
    )
    return features.reset_index()


def within_stage_z(frame, columns):
    """Z-score each property inside its stage, so 'which stage' cannot do the explaining."""
    out = frame.copy()
    for column in columns:
        grouped = out.groupby("stage")[column]
        out[column] = (out[column] - grouped.transform("mean")) / grouped.transform("std").replace(0, np.nan)
    return out


def rank_auc(values, labels):
    """P(random positive ranks above random negative); ties count a half. 0.5 = nothing."""
    positive, negative = values[labels], values[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return np.nan
    order = stats.rankdata(np.concatenate([positive, negative]))
    return float((order[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def property_stats(features, sweep, properties, group_size):
    """Per (dataset, property): Spearman rho and top-vs-bottom AUC, raw and within-stage."""
    raw = features
    zed = within_stage_z(features, properties)

    records = []
    for dataset, group in sweep.groupby("dataset"):
        effects = group.set_index("pruning_id")["effect"]
        helpful = set(effects.nlargest(group_size).index)
        harmful = set(effects.nsmallest(group_size).index)

        for scope, table in (("raw", raw), ("within_stage", zed)):
            merged = table.set_index("pruning_id").join(effects.rename("effect")).dropna(subset=["effect"])
            labels = np.array([head in helpful for head in merged.index])
            extremes = np.array([head in helpful or head in harmful for head in merged.index])

            for prop in properties:
                values = merged[prop].to_numpy()
                if np.all(np.isnan(values)) or np.nanstd(values) == 0:
                    continue
                rho, pval = stats.spearmanr(values, merged["effect"].to_numpy(), nan_policy="omit")
                records.append(
                    {
                        "dataset": dataset,
                        "scope": scope,
                        "property": prop,
                        "rho": float(rho),
                        "p_value": float(pval),
                        "auc": rank_auc(values[extremes], labels[extremes]),
                        "n_heads": int(len(merged)),
                    }
                )
    return pd.DataFrame(records)


def mean_effect_stats(features, sweep, properties):
    """The same test against the effect *averaged over datasets*.

    Per-dataset effects are single noisy measurements, which attenuates every correlation
    toward zero. Averaging the seven datasets raises the signal-to-noise of the target at
    the cost of the per-dataset detail, so this is the view with the most power — and the
    only one where anything reaches significance. p-values are Bonferroni-corrected for
    the number of properties tested, since this is a screen over many candidates.
    """
    mean_effect = sweep.groupby("pruning_id")["effect"].mean().rename("effect")
    merged = features.set_index("pruning_id").join(mean_effect).reset_index()
    zed = within_stage_z(merged, properties)

    records = []
    for scope, table in (("raw", merged), ("within_stage", zed)):
        for prop in properties:
            rho, pval = stats.spearmanr(table[prop], table["effect"], nan_policy="omit")
            records.append(
                {
                    "scope": scope,
                    "property": prop,
                    "rho": float(rho),
                    "p_value": float(pval),
                    "p_bonferroni": min(1.0, float(pval) * len(properties)),
                    "n_heads": int(len(table)),
                }
            )
    frame = pd.DataFrame(records)
    frame["significant_corrected"] = frame["p_bonferroni"] < 0.05
    return frame


def summarise(stats_frame):
    """Mean over datasets, plus how many datasets agree on the sign — the honest headline."""
    summary = (
        stats_frame.groupby(["scope", "property"])
        .agg(
            mean_rho=("rho", "mean"),
            min_rho=("rho", "min"),
            max_rho=("rho", "max"),
            mean_auc=("auc", "mean"),
            datasets=("rho", "size"),
            same_sign=("rho", lambda values: int(max((values > 0).sum(), (values < 0).sum()))),
        )
        .reset_index()
    )
    summary["consistent"] = summary["same_sign"] == summary["datasets"]
    return summary.sort_values(["scope", "mean_rho"], key=lambda s: s.abs() if s.name == "mean_rho" else s,
                               ascending=[True, False])


# --------------------------------------------------------------------------- figures


def fig_property_ranking(summary, out_dir, formats):
    """Which properties separate the two groups at all — within stage vs raw."""
    scopes = ["within_stage", "raw"]
    order = (
        summary[summary["scope"] == "within_stage"]
        .reindex(summary[summary["scope"] == "within_stage"]["mean_rho"].abs().sort_values().index)["property"]
        .tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.34 * len(order) + 2.4), sharey=True)
    for ax, scope in zip(axes, scopes):
        subset = summary[summary["scope"] == scope].set_index("property").reindex(order)
        ypos = np.arange(len(order))
        colors = [HELPFUL_COLOR if value > 0 else HARMFUL_COLOR for value in subset["mean_rho"]]
        ax.barh(ypos, subset["mean_rho"].to_numpy(), height=0.6, color=colors, zorder=2)
        ax.hlines(ypos, subset["min_rho"].to_numpy(), subset["max_rho"].to_numpy(),
                  color=INK_MUTED, linewidth=1.4, zorder=3)
        for y, consistent in zip(ypos, subset["consistent"]):
            if consistent:
                ax.text(0.985, y, "all 7 agree", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=7.5, color=INK_MUTED)
        ax.axvline(0, color=AXIS, linewidth=1.0)
        ax.set_yticks(ypos, [PROPERTY_LABELS.get(name, name) for name in order], fontsize=8.5)
        ax.set_xlabel("Spearman ρ with pruning effect")
        ax.set_title(
            "within stage (confound removed)" if scope == "within_stage" else "raw (stage confound included)",
            loc="left", fontsize=10,
        )
        style_axes(ax, grid_axis="x")

    fig.legend(
        handles=[
            Line2D([], [], color=HELPFUL_COLOR, linewidth=6, label="higher → removal helps more"),
            Line2D([], [], color=HARMFUL_COLOR, linewidth=6, label="higher → removal hurts more"),
            Line2D([], [], color=INK_MUTED, linewidth=1.6, label="range over the 7 datasets"),
        ],
        loc="lower left", bbox_to_anchor=(0.005, -0.02), ncol=3, fontsize=9,
        labelcolor=INK_SECONDARY, frameon=False,
    )
    fig.suptitle("Which head properties predict that removing the head helps?",
                 x=0.005, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.005, 0.945, "Mean Spearman ρ over 7 datasets, 184 heads each",
             ha="left", va="top", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    save(fig, out_dir, "01_property_ranking", formats)


def fig_generality_scatter(features, sweep, out_dir, formats,
                           shown=("general_share", "contrib_ratio")):
    """The decisive pair: is it *how loud* the head is, or *what it is loud on*?

    Left: overall activation magnitude. Right: the general-vs-medical contrast. If the
    story were "general-audio heads conflict with the medical task", the right panel is
    where it would show up.
    """
    mean_effect = sweep.groupby("pruning_id")["effect"].mean().rename("effect")
    merged = features.set_index("pruning_id").join(mean_effect).reset_index()
    zed = within_stage_z(merged, list(shown))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for ax, top_property in zip(axes, shown):
        table = zed
        x = table[top_property].to_numpy()
        y = table["effect"].to_numpy()
        rho, pval = stats.spearmanr(x, y, nan_policy="omit")

        for stage, marker in zip(sorted(table["stage"].unique()), ("o", "s", "^", "D")):
            mask = table["stage"] == stage
            ax.scatter(x[mask], y[mask], s=26, marker=marker, color=SERIES, alpha=0.6,
                       linewidths=0, label=f"stage {stage}")

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() > 2:
            slope, intercept = np.polyfit(x[finite], y[finite], 1)
            span = np.linspace(np.nanmin(x), np.nanmax(x), 2)
            ax.plot(span, slope * span + intercept, color=INK_MUTED, linewidth=1.4)

        ax.axhline(0, color=AXIS, linewidth=1.0)
        ax.set_xlabel(PROPERTY_LABELS.get(top_property, top_property) + " (z within stage)")
        ax.set_ylabel("mean ΔAUROC from removing the head")
        ax.set_title(f"ρ={rho:+.2f}, p={pval:.1g}", loc="left", fontsize=10)
        style_axes(ax, grid_axis="both")
    axes[0].legend(fontsize=8.5, labelcolor=INK_SECONDARY, loc="best")

    fig.suptitle("Is it how loud the head is, or what it is loud on?",
                 x=0.005, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.005, 0.945,
             "One point per head · effect averaged over the 7 medical datasets · "
             "properties z-scored within stage",
             ha="left", va="top", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, out_dir, "02_generality_vs_effect", formats)


def fig_group_profiles(features, sweep, out_dir, formats, group_size, properties):
    """Helpful vs harmful groups, property by property, as within-stage z-scores."""
    mean_effect = sweep.groupby("pruning_id")["effect"].mean()
    helpful = set(mean_effect.nlargest(group_size).index)
    harmful = set(mean_effect.nsmallest(group_size).index)

    zed = within_stage_z(features, properties).set_index("pruning_id")
    groups = {
        "removal helps": zed.loc[[h for h in zed.index if h in helpful]],
        "removal hurts": zed.loc[[h for h in zed.index if h in harmful]],
    }

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(properties) + 2.4))
    ypos = np.arange(len(properties))
    for (name, table), offset, color in zip(
        groups.items(), (-0.18, 0.18), (HELPFUL_COLOR, HARMFUL_COLOR)
    ):
        means = [table[prop].mean() for prop in properties]
        errors = [table[prop].std() / np.sqrt(max(len(table), 1)) for prop in properties]
        ax.barh(ypos + offset, means, height=0.32, color=color, label=f"{name} (n={len(table)})", zorder=2)
        ax.hlines(ypos + offset, np.array(means) - np.array(errors), np.array(means) + np.array(errors),
                  color=INK_MUTED, linewidth=1.2, zorder=3)

    ax.axvline(0, color=AXIS, linewidth=1.0)
    ax.set_yticks(ypos, [PROPERTY_LABELS.get(name, name) for name in properties], fontsize=8.5)
    ax.set_xlabel("mean z-score within stage (0 = the stage's average head)")
    titles(
        ax, f"Profile of the top-{group_size} heads on each side",
        "Ranked by mean effect over the 7 datasets · error bars are standard errors",
    )
    ax.legend(fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")
    style_axes(ax, grid_axis="x")
    save(fig, out_dir, "03_group_profiles", formats)


def fig_stage_confound(features, sweep, out_dir, formats):
    """How much of everything is just 'which stage' — the reason for the within-stage view."""
    mean_effect = sweep.groupby("pruning_id")["effect"].mean().rename("effect")
    merged = features.set_index("pruning_id").join(mean_effect).reset_index()

    columns = ["generality", "general_share", "w_norm", "ov_spectral", "effect"]
    fig, axes = plt.subplots(1, len(columns), figsize=(3.0 * len(columns), 3.6), sharex=True)
    for ax, column in zip(axes, columns):
        stages = sorted(merged["stage"].unique())
        data = [merged.loc[merged["stage"] == stage, column].dropna().to_numpy() for stage in stages]
        parts = ax.boxplot(data, positions=np.arange(len(stages)), widths=0.55, patch_artist=True,
                           showfliers=False, medianprops=dict(color=INK, linewidth=1.3),
                           whiskerprops=dict(color=AXIS), capprops=dict(color=AXIS))
        for patch in parts["boxes"]:
            patch.set_facecolor(SERIES_SOFT)
            patch.set_edgecolor(AXIS)
            patch.set_linewidth(0.8)
        ax.set_xticks(np.arange(len(stages)), [f"s{stage}" for stage in stages], fontsize=8.5)
        ax.set_title(PROPERTY_LABELS.get(column, column), loc="left", fontsize=9)
        style_axes(ax, grid_axis="y")
    axes[0].set_ylabel("value")

    fig.suptitle("Everything varies by stage — which is why the statistics are computed within stage",
                 x=0.005, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, out_dir, "04_stage_confound", formats)


# ------------------------------------------------- selected heads vs. everything else


def per_dataset_features(static, activations, sweep, k):
    """One row per (dataset, head): properties + which of the three groups it falls in.

    The unit is (dataset, head), not head, for two reasons: "helpful to remove" is a
    per-dataset property, and the medical activation statistics are per-corpus, so each
    dataset gets its own. Two heads are in fact selected as helpful for one dataset and
    harmful for another, which a per-head grouping could not represent.
    """
    general = (
        activations[activations["domain"] == "general"]
        .groupby("pruning_id")[["contrib_rms", "ctx_rms", "energy_share"]]
        .mean()
        .add_prefix("general_")
    )
    medical = activations[activations["domain"] == "medical"].rename(
        columns={"corpus": "dataset"}
    )[["dataset", "pruning_id", "contrib_rms", "ctx_rms", "energy_share"]]
    medical = medical.rename(
        columns={
            "contrib_rms": "medical_contrib_rms",
            "ctx_rms": "medical_ctx_rms",
            "energy_share": "medical_energy_share",
        }
    )

    frame = (
        sweep[["dataset", "pruning_id", "effect"]]
        .merge(medical, on=["dataset", "pruning_id"], how="inner")
        .merge(general.reset_index(), on="pruning_id", how="left")
        .merge(static, on="pruning_id", how="left")
    )
    frame["generality"] = frame["general_energy_share"] - frame["medical_energy_share"]
    frame["contrib_ratio"] = np.log2(frame["general_contrib_rms"] / frame["medical_contrib_rms"])
    frame = frame.rename(
        columns={"medical_energy_share": "medical_share", "general_energy_share": "general_share"}
    )

    # Same selection rule as the control experiment: top-k of each sign, per dataset.
    frame["group"] = "rest"
    for dataset, group in frame.groupby("dataset"):
        helps = group[group["effect"] > 0].nlargest(k, "effect").index
        hurts = group[group["effect"] < 0].nsmallest(k, "effect").index
        frame.loc[helps, "group"] = "helps"
        frame.loc[hurts, "group"] = "hurts"
    return frame


def zscore_within(frame, properties, keys):
    """Z-score inside (dataset, stage), so neither the corpus nor the stage explains it."""
    out = frame.copy()
    for column in properties:
        grouped = out.groupby(keys)[column]
        spread = grouped.transform("std").replace(0, np.nan)
        out[column] = (out[column] - grouped.transform("mean")) / spread
    return out


def group_vs_rest(frame, properties, n_permutations, seed=0):
    """Is each selected group distinct from the ordinary heads, property by property?

    The null is "this property is unrelated to the pruning effect": for each draw, pick a
    random k heads per dataset instead of the top-k, and recompute the same statistic.
    That respects the design — same group sizes, same datasets, same stage mix on average
    — and needs no independence assumption, which matters because the seven datasets share
    the same 184 heads.
    """
    rng = np.random.default_rng(seed)
    zed = zscore_within(frame, properties, ["dataset", "stage"])
    datasets = sorted(zed["dataset"].unique())
    # "either" pools both signs: it asks whether a head that matters *at all* differs
    # from one that does not, which is a different question from which way it matters.
    members = {
        "helps": zed["group"] == "helps",
        "hurts": zed["group"] == "hurts",
        "either": zed["group"].isin(["helps", "hurts"]),
    }
    sizes = {
        (dataset, group): int(((zed["dataset"] == dataset) & mask).sum())
        for dataset in datasets
        for group, mask in members.items()
    }

    records = []
    for prop in properties:
        column = zed[prop].to_numpy()
        by_dataset = {d: np.flatnonzero((zed["dataset"] == d).to_numpy()) for d in datasets}

        for group, membership in members.items():
            mask = membership.to_numpy()
            selected = column[mask]
            others = column[(zed["group"] == "rest").to_numpy()]
            if len(selected) == 0 or np.all(np.isnan(selected)):
                continue

            observed = float(np.nanmedian(selected))
            delta = cliffs_delta(selected, others)

            # Null distribution of the same median, over random selections of equal size.
            draws = np.empty(n_permutations)
            for i in range(n_permutations):
                picked = np.concatenate(
                    [
                        rng.choice(by_dataset[d], size=sizes[(d, group)], replace=False)
                        for d in datasets
                        if sizes[(d, group)]
                    ]
                )
                draws[i] = np.nanmedian(column[picked])
            # Two-sided: how often is a random selection at least this extreme?
            centre = np.nanmedian(draws)
            p_value = float(
                (np.abs(draws - centre) >= abs(observed - centre)).sum() + 1
            ) / (n_permutations + 1)

            records.append(
                {
                    "property": prop,
                    "group": group,
                    "n": int(np.isfinite(selected).sum()),
                    "median_z": observed,
                    "null_median_z": float(centre),
                    "null_p05": float(np.nanpercentile(draws, 2.5)),
                    "null_p95": float(np.nanpercentile(draws, 97.5)),
                    "cliffs_delta": delta,
                    "p_permutation": p_value,
                }
            )

    result = pd.DataFrame(records)
    result["p_bonferroni"] = (result["p_permutation"] * len(result)).clip(upper=1.0)
    result["significant"] = result["p_bonferroni"] < 0.05
    return result.sort_values("p_permutation")


def cliffs_delta(a, b):
    """P(a > b) - P(a < b): +1 = every selected head above every ordinary one, 0 = no shift."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    greater = float(sum((a[:, None] > b[None, :]).sum(axis=1)))
    less = float(sum((a[:, None] < b[None, :]).sum(axis=1)))
    return (greater - less) / (len(a) * len(b))


def fig_group_vs_rest(contrast, frame, properties, out_dir, formats, k):
    """Each selected group against the ordinary heads, with the random-selection band."""
    order = (
        contrast.groupby("property")["median_z"].apply(lambda v: v.abs().max())
        .sort_values().index.tolist()
    )
    groups = [g for g in ("helps", "hurts", "either") if g in set(contrast["group"])]
    titles_by_group = {
        "helps": f"top-{k} heads whose removal HELPED",
        "hurts": f"top-{k} heads whose removal HURT",
        "either": "both groups pooled — heads that matter at all",
    }
    fig, axes = plt.subplots(1, len(groups), figsize=(6.0 * len(groups), 0.34 * len(order) + 2.6),
                             sharey=True)

    for ax, group, color in zip(np.atleast_1d(axes), groups,
                                (HELPFUL_COLOR, HARMFUL_COLOR, SERIES)):
        subset = contrast[contrast["group"] == group].set_index("property").reindex(order)
        ypos = np.arange(len(order))
        # What a random selection of the same size does — anything inside this band is
        # indistinguishable from an ordinary head.
        ax.barh(
            ypos, subset["null_p95"] - subset["null_p05"], left=subset["null_p05"],
            height=0.72, color=GRID, zorder=1,
        )
        ax.scatter(subset["median_z"], ypos, s=46, color=color, zorder=3)
        for y, (value, marked) in enumerate(zip(subset["median_z"], subset["significant"])):
            if marked:
                ax.annotate("*", (value, y), textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=13, color=INK)
        ax.axvline(0, color=AXIS, linewidth=1.0)
        ax.set_yticks(ypos, [PROPERTY_LABELS.get(name, name) for name in order], fontsize=8.5)
        ax.set_xlabel("median z-score within (dataset, stage)")
        ax.set_title(f"{titles_by_group[group]}  (n={int(subset['n'].max())})",
                     loc="left", fontsize=10)
        style_axes(ax, grid_axis="x")

    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=INK_SECONDARY, markersize=7,
                   label="selected group median (coloured per panel)"),
            Line2D([], [], color=GRID, linewidth=8, label="95% of random same-size selections"),
            Line2D([], [], marker="$*$", linestyle="none", color=INK, markersize=9,
                   label="significant after Bonferroni"),
        ],
        loc="lower left", bbox_to_anchor=(0.005, -0.02), ncol=3, fontsize=9,
        labelcolor=INK_SECONDARY, frameon=False,
    )
    fig.suptitle("Are the selected heads distinguishable from ordinary heads?",
                 x=0.005, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.005, 0.945,
             "0 = the average head of the same dataset and stage · a dot inside the grey band "
             "is what chance already produces",
             ha="left", va="top", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    save(fig, out_dir, "05_selected_vs_rest", formats)


# ------------------------------------------------------------------------------- run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--pooling", default="projected")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select-k", type=int, default=3,
                        help="Heads per sign per dataset for the selected-vs-rest contrast.")
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--group-size", type=int, default=20,
                        help="Heads per side when forming the helpful/harmful contrast groups.")
    parser.add_argument("--formats", nargs="+", default=["png"])
    args = parser.parse_args()

    static, activations, sweep = load(args.results_dir, args.pooling, args.seed)
    features = head_features(static, activations)
    print(f"{len(features)} heads x {sweep['dataset'].nunique()} datasets")
    print(f"corpora probed: {sorted(activations['corpus'].unique())}")

    properties = [
        "generality", "general_share", "medical_share", "contrib_ratio",
        "general_contrib_rms", "medical_contrib_rms",
        "w_norm", "w_norm_qkv", "w_norm_out", "w_rms",
        "w_kurtosis", "w_participation", "w_max_over_rms", "ov_spectral",
    ]
    properties = [p for p in properties if p in features.columns]

    stats_frame = property_stats(features, sweep, properties, args.group_size)
    summary = summarise(stats_frame)
    mean_stats = mean_effect_stats(features, sweep, properties)

    out_dir = Path(args.figures_dir) / f"head_properties_{args.pooling}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "head_features.csv", index=False, float_format="%.6g")
    stats_frame.to_csv(out_dir / "property_stats_per_dataset.csv", index=False, float_format="%.6g")
    summary.to_csv(out_dir / "property_summary.csv", index=False, float_format="%.6g")
    mean_stats.to_csv(out_dir / "property_stats_mean_effect.csv", index=False, float_format="%.6g")

    # How much do general and medical activation profiles differ at all? If they barely
    # do, there are no "general-audio specialist" heads for the contrast to find.
    agreement = stats.pearsonr(features["general_share"], features["medical_share"])[0]
    print(f"\ngeneral vs medical energy share across heads: pearson r = {agreement:.3f}")

    print("\nWithin-stage (the confound-free view), ranked by |mean rho|:")
    view = summary[summary["scope"] == "within_stage"].copy()
    view["property"] = view["property"].map(lambda name: PROPERTY_LABELS.get(name, name))
    print(view[["property", "mean_rho", "min_rho", "max_rho", "mean_auc", "consistent"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print("\nAgainst the MEAN effect (highest-power view), within stage, |rho| ranked:")
    view = mean_stats[mean_stats["scope"] == "within_stage"].copy()
    view = view.reindex(view["rho"].abs().sort_values(ascending=False).index)
    view["property"] = view["property"].map(lambda name: PROPERTY_LABELS.get(name, name))
    print(view[["property", "rho", "p_value", "p_bonferroni", "significant_corrected"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    per_dataset = per_dataset_features(static, activations, sweep, args.select_k)
    contrast = group_vs_rest(per_dataset, properties, args.permutations)
    contrast.to_csv(out_dir / "selected_vs_rest.csv", index=False, float_format="%.6g")
    counts = per_dataset["group"].value_counts()
    print(f"\nSelected-vs-rest (top-{args.select_k} per sign per dataset): "
          f"{counts.get('helps', 0)} helps, {counts.get('hurts', 0)} hurts, {counts.get('rest', 0)} rest "
          f"(dataset, head) rows")
    print(contrast[["property", "group", "median_z", "cliffs_delta", "p_permutation",
                    "p_bonferroni", "significant"]]
          .head(10).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    fig_property_ranking(summary, out_dir, args.formats)
    fig_group_vs_rest(contrast, per_dataset, properties, out_dir, args.formats, args.select_k)
    fig_generality_scatter(features, sweep, out_dir, args.formats)
    fig_group_profiles(features, sweep, out_dir, args.formats, args.group_size, properties)
    fig_stage_confound(features, sweep, out_dir, args.formats)
    print("\nDone.")


if __name__ == "__main__":
    main()
