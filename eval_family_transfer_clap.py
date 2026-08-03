"""Held-out test: are the heads worth removing shared within an acoustic task family?

The cross-task correlation matrix (``analyze_transfer.py``) found no family block
structure, but it is a low-power statistic: it weights all 184 heads equally and most of
them sit at the noise floor. This experiment asks the same question with the highest
signal-to-noise design available — select a *set* of heads on other datasets, remove them
all at once, and measure the AUROC change directly on a dataset that was never used to
choose them.

For each target dataset D, three head sets of size k are selected, none of them using D:

    own_family     top-k by mean effect over the other datasets of D's family
    other_family   top-k by mean effect over the datasets of the *other* family
    universal      top-k by mean effect over every dataset except D
    random         k heads drawn at random (n trials, the floor)

All k heads are removed simultaneously, and every comparison is *within* the same target
dataset. That matters: the transfer matrix showed a large per-target main effect (ICBHI
gains from almost any transferred head, CirCor loses from almost any), which would swamp
a between-target comparison.

Prediction if head-harmfulness is organised by family: own_family > other_family on the
held-out target. If harmfulness is a model-wide property: universal wins everywhere and
own ~ other. If it is per-(head, dataset) noise: all three collapse onto random.

Usage:
    python eval_family_transfer_clap.py
    python eval_family_transfer_clap.py --k 10 --random-trials 10
    python eval_family_transfer_clap.py --resume
"""

import argparse
import csv
import hashlib
import random
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from eval_knn_clap import ClapBackbone
from eval_prune_clap import cache_split, encode_cached, prune_units
from knn_eval_core import build_dataset, knn_evaluate

# Anatomy, fixed independently of any result: BMD (Mit/Pul/Aor/Tri), CirCor (MV/AV/PV/TV),
# CinC (PhysioNet heart sounds) and ZCHSound (congenital heart disease) are auscultation
# recordings of the heart; ICBHI, KAUH and SPRSound are lung-sound recordings.
FAMILIES = {
    "BMD": "cardiac",
    "CinC": "cardiac",
    "CirCor": "cardiac",
    "ZCHSound": "cardiac",
    "ICBHI": "respiratory",
    "KAUH": "respiratory",
    "SPRSound": "respiratory",
}

CSV_FIELDS = [
    "dataset",
    "family",
    "seed",
    "pooling",
    "condition",
    "trial",
    "k",
    "selected_on",
    "head_set",
    "AUROC",
]


def effect_matrix(results_dir, pooling, seed):
    sweep = pd.read_csv(Path(results_dir) / f"prune_clap_head_{pooling}_seed{seed}.csv")
    baseline = pd.read_csv(Path(results_dir) / f"prune_clap_none_{pooling}_seed{seed}.csv")
    sweep["effect"] = sweep["AUROC"] - sweep["dataset"].map(baseline.set_index("dataset")["AUROC"])
    return sweep.pivot_table(index="pruning_id", columns="dataset", values="effect").sort_index()


def head_sets(matrix, target, k, random_trials, seed):
    """The head sets to remove for one target, none of them selected using that target."""
    others = [d for d in matrix.columns if d != target]
    own = [d for d in others if FAMILIES.get(d) == FAMILIES.get(target)]
    cross = [d for d in others if FAMILIES.get(d) != FAMILIES.get(target)]

    def top_k(sources):
        if not sources:
            return None
        return list(matrix[sources].mean(axis=1).nlargest(k).index), sources

    sets = {}
    for name, sources in (("own_family", own), ("other_family", cross), ("universal", others)):
        picked = top_k(sources)
        if picked is not None:
            sets[(name, 0)] = picked

    # The floor: whatever removing k arbitrary heads does to this dataset.
    heads = list(matrix.index)
    for trial in range(1, random_trials + 1):
        key = f"{seed}|{target}|random|{trial}".encode()
        draw_seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
        rng = np.random.default_rng(draw_seed)
        sets[("random", trial)] = (
            sorted(rng.choice(heads, size=k, replace=False).tolist()),
            [],
        )
    return sets


@contextmanager
def removed(pruners, head_ids):
    """Remove every head in the set at once, then restore all of them."""
    with ExitStack() as stack:
        for head_id in head_ids:
            stack.enter_context(pruners[head_id]())
        yield


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="label-mapping.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--model", default=ClapBackbone.default_model_id)
    parser.add_argument("--pooling", choices=list(ClapBackbone.pooling_choices), default="projected")
    parser.add_argument("--k", type=int, default=10, help="Heads removed jointly per set.")
    parser.add_argument("--random-trials", type=int, default=10)
    parser.add_argument("-k2", "--knn-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-seconds", type=int, default=ClapBackbone.default_clip_seconds)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def open_csv(path, resume):
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if resume and path.exists():
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                done.add((row["dataset"], row["condition"], int(row["trial"])))
        out = open(path, "a", newline="")
    else:
        out = open(path, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
    if not done:
        writer.writeheader()
    return out, writer, done


def main():
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    matrix = effect_matrix(args.results_dir, args.pooling, args.seed)
    print(f"Head-effect matrix: {matrix.shape[0]} heads x {matrix.shape[1]} datasets")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} (pooling={args.pooling})")
    backbone = ClapBackbone(args.model, args.pooling, device, args.clip_seconds)
    pruners = dict(prune_units(backbone.model, "head"))

    entries = {e["name"]: e for e in yaml.safe_load(open(args.config))["datasets"]}
    targets = [d for d in matrix.columns if not args.datasets or d in args.datasets]

    output = Path(args.output or f"results/family_transfer_{args.pooling}_seed{args.seed}.csv")
    handle, writer, done = open_csv(output, args.resume)
    if done:
        print(f"Resuming {output}: {len(done)} row(s) already present")

    try:
        for target in targets:
            entry = entries.get(target)
            if entry is None:
                print(f"\n=== {target} ===\n  skipped: not in {args.config}")
                continue

            sets = head_sets(matrix, target, args.k, args.random_trials, args.seed)
            pending = [("intact", 0)] + list(sets.keys())
            pending = [item for item in pending if (target, item[0], item[1]) not in done]
            if not pending:
                print(f"\n=== {target} ===\n  skipped: already complete")
                continue

            print(f"\n=== {target} ({FAMILIES.get(target, '?')}) ===", flush=True)
            for name in ("own_family", "other_family", "universal"):
                if (name, 0) in sets:
                    print(f"  {name:<13} selected on {sets[(name, 0)][1]}")
                    print(f"                {', '.join(sets[(name, 0)][0])}")

            train_set = build_dataset(
                entry["path"], entry["label_map"], "train", args.clip_seconds, backbone.sample_rate
            )
            test_set = build_dataset(
                entry["path"], entry["label_map"], "test", args.clip_seconds, backbone.sample_rate
            )
            train_batches, train_y = cache_split(
                train_set, backbone, args.batch_size, args.num_workers, f"{target} train (cache)"
            )
            test_batches, test_y = cache_split(
                test_set, backbone, args.batch_size, args.num_workers, f"{target} test (cache)"
            )

            for condition, trial in pending:
                if condition == "intact":
                    heads, sources = [], []
                else:
                    heads, sources = sets[(condition, trial)]

                with removed(pruners, heads):
                    train_x = encode_cached(train_batches, backbone)
                    test_x = encode_cached(test_batches, backbone)

                auroc = knn_evaluate(train_x, train_y, test_x, test_y, args.knn_k)["auroc"]
                writer.writerow(
                    {
                        "dataset": target,
                        "family": FAMILIES.get(target, ""),
                        "seed": args.seed,
                        "pooling": args.pooling,
                        "condition": condition,
                        "trial": trial,
                        "k": 0 if condition == "intact" else args.k,
                        "selected_on": ";".join(sources),
                        "head_set": ";".join(heads),
                        "AUROC": f"{auroc:.6f}",
                    }
                )
                handle.flush()
                print(f"  {condition:<13} t{trial:<3} auroc={auroc:.4f}", flush=True)

            del train_batches, test_batches
    finally:
        handle.close()

    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
