"""Error bars for the out-of-sample damage done by the "hurts" heads.

The split-stability experiment showed that heads whose removal *hurts* keep hurting when
the two splits swap roles. But each dataset's out-of-sample number came from a single
alternative split, so a reviewer can reasonably ask whether SPRSound and CirCor were a
lucky second draw. This turns "survives one swap" into "survives with an error bar".

Design, deliberately simple:

* Heads are chosen on the **forward** split only (fit on train, score on test) — the same
  top-k "hurts" selection as before. Nothing about the scored recordings informs it.
* Damage is measured in the **reverse** direction (fit on test, score on train), which is
  the out-of-sample readout.
* The scored set is resampled with replacement, ``--resamples`` times. The fitted
  classifier is untouched, so per-recording scores are computed once and the bootstrap
  only changes which recordings enter the AUROC — this is re-scoring, not re-embedding.
* Every condition sees the **same** resample indices, so the intact model and each pruned
  model are compared on identical recordings and the interval is on the *difference*.
* ``--random-heads`` heads are carried through the identical pipeline as the floor: what a
  arbitrary head does to the same dataset under the same resamples.

No stratification, no new splits, no re-embedding beyond the one forward pass each
configuration needs.

Usage:
    python eval_bootstrap_hurts.py
    python eval_bootstrap_hurts.py --datasets SPRSound CirCor --resamples 200
"""

import argparse
import csv
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from eval_knn_clap import ClapBackbone
from eval_prune_clap import cache_split, encode_cached, prune_units
from knn_eval_core import build_dataset

# The datasets whose "hurts" side replicated across the split swap at p < 0.05.
REPLICATED = ["BMD", "CirCor", "KAUH", "SPRSound"]

CSV_FIELDS = [
    "dataset",
    "seed",
    "pooling",
    "condition",
    "pruning_id",
    "forward_effect",
    "resample",
    "auroc_intact",
    "auroc_pruned",
    "effect",
]


def knn_scores(fit_x, fit_y, score_x, k):
    """Positive-class score per scored recording — the same classifier as ``knn_evaluate``."""
    fit_x = fit_x / np.clip(np.linalg.norm(fit_x, axis=1, keepdims=True), 1e-12, None)
    score_x = score_x / np.clip(np.linalg.norm(score_x, axis=1, keepdims=True), 1e-12, None)

    knn = KNeighborsClassifier(
        n_neighbors=min(k, len(fit_x)), metric="cosine", weights="distance"
    )
    knn.fit(fit_x, fit_y)
    proba = knn.predict_proba(score_x)
    positive = list(knn.classes_).index(1) if 1 in knn.classes_ else None
    return proba[:, positive] if positive is not None else np.zeros(len(score_x))


def resample_indices(labels, n_resamples, seed):
    """Bootstrap draws of the scored set; a draw with one class has no AUROC and is dropped."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        for _attempt in range(20):
            index = rng.integers(0, len(labels), size=len(labels))
            if len(np.unique(labels[index])) == 2:
                draws.append(index)
                break
    return draws


def select_hurts(results_dir, pooling, seed, k):
    """Top-k heads whose removal hurt most on the forward split, per dataset."""
    sweep = pd.read_csv(Path(results_dir) / f"prune_clap_head_{pooling}_seed{seed}.csv")
    baseline = pd.read_csv(Path(results_dir) / f"prune_clap_none_{pooling}_seed{seed}.csv")
    sweep["effect"] = sweep["AUROC"] - sweep["dataset"].map(baseline.set_index("dataset")["AUROC"])
    chosen = {}
    for dataset, group in sweep.groupby("dataset"):
        picked = group.nsmallest(k, "effect")
        chosen[dataset] = list(zip(picked["pruning_id"], picked["effect"]))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="label-mapping.yaml")
    parser.add_argument("--datasets", nargs="*", default=REPLICATED)
    parser.add_argument("--model", default=ClapBackbone.default_model_id)
    parser.add_argument("--pooling", choices=list(ClapBackbone.pooling_choices), default="projected")
    parser.add_argument("--top-k", type=int, default=10, help="'Hurts' heads carried forward.")
    parser.add_argument("--random-heads", type=int, default=10, help="Floor: arbitrary heads.")
    parser.add_argument("--resamples", type=int, default=50)
    parser.add_argument("-k", "--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-seconds", type=int, default=ClapBackbone.default_clip_seconds)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hurts = select_hurts(args.results_dir, args.pooling, args.seed, args.top_k)
    entries = {e["name"]: e for e in yaml.safe_load(open(args.config))["datasets"]}
    targets = [d for d in args.datasets if d in hurts]
    if not targets:
        raise SystemExit(f"No usable datasets in {args.datasets}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} (pooling={args.pooling})")
    backbone = ClapBackbone(args.model, args.pooling, device, args.clip_seconds)
    pruners = dict(prune_units(backbone.model, "head"))
    all_heads = sorted(pruners)

    output = Path(args.output or f"results/bootstrap_hurts_{args.pooling}_seed{args.seed}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = open(output, "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()

    try:
        for dataset in targets:
            entry = entries[dataset]
            print(f"\n=== {dataset} ===", flush=True)

            train_set = build_dataset(
                entry["path"], entry["label_map"], "train", args.clip_seconds, backbone.sample_rate
            )
            test_set = build_dataset(
                entry["path"], entry["label_map"], "test", args.clip_seconds, backbone.sample_rate
            )
            train_batches, train_y = cache_split(
                train_set, backbone, args.batch_size, args.num_workers, f"{dataset} train (cache)"
            )
            test_batches, test_y = cache_split(
                test_set, backbone, args.batch_size, args.num_workers, f"{dataset} test (cache)"
            )

            # Reverse direction: the classifier is fitted on test, the train split is scored.
            fit_batches, fit_y = test_batches, test_y
            score_batches, score_y = train_batches, train_y
            draws = resample_indices(score_y, args.resamples, args.seed)
            print(f"  fit on {len(fit_y)} (test), score {len(score_y)} (train), "
                  f"{len(draws)} usable resamples", flush=True)

            intact_scores = knn_scores(
                encode_cached(fit_batches, backbone),
                fit_y,
                encode_cached(score_batches, backbone),
                args.k,
            )
            intact_auroc = np.array(
                [roc_auc_score(score_y[index], intact_scores[index]) for index in draws]
            )
            print(f"  intact reverse AUROC {intact_auroc.mean():.4f} "
                  f"[{np.percentile(intact_auroc, 2.5):.4f}, {np.percentile(intact_auroc, 97.5):.4f}]",
                  flush=True)

            # Deterministic across processes; Python's hash() is salted per run.
            digest = int.from_bytes(hashlib.sha256(dataset.encode()).digest()[:4], "big")
            rng = np.random.default_rng(args.seed + digest % 1000)
            random_heads = rng.choice(
                [h for h in all_heads if h not in {i for i, _ in hurts[dataset]}],
                size=args.random_heads, replace=False,
            )
            conditions = [("hurts", head, effect) for head, effect in hurts[dataset]]
            conditions += [("random", head, float("nan")) for head in random_heads]

            for condition, head_id, forward_effect in conditions:
                with pruners[head_id]():
                    scores = knn_scores(
                        encode_cached(fit_batches, backbone),
                        fit_y,
                        encode_cached(score_batches, backbone),
                        args.k,
                    )
                # Same resample indices as the intact model: the difference is paired.
                effects = []
                for resample, index in enumerate(draws):
                    pruned = roc_auc_score(score_y[index], scores[index])
                    effects.append(pruned - intact_auroc[resample])
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "seed": args.seed,
                            "pooling": args.pooling,
                            "condition": condition,
                            "pruning_id": head_id,
                            "forward_effect": "" if np.isnan(forward_effect) else f"{forward_effect:.6f}",
                            "resample": resample,
                            "auroc_intact": f"{intact_auroc[resample]:.6f}",
                            "auroc_pruned": f"{pruned:.6f}",
                            "effect": f"{pruned - intact_auroc[resample]:.6f}",
                        }
                    )
                handle.flush()
                print(f"  {condition:<7} {head_id:<12} effect {np.mean(effects):+.4f} "
                      f"[{np.percentile(effects, 2.5):+.4f}, {np.percentile(effects, 97.5):+.4f}]",
                      flush=True)

            del train_batches, test_batches
    finally:
        handle.close()

    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
