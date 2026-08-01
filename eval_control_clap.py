"""Control experiment: is a head-pruning gain a real arrangement effect, or just regularisation?

The head sweep (``eval_prune_clap.py``) found heads whose *removal* raises k-NN AUROC on
a given dataset. Removing a head does two things at once: it destroys that head's
specific computation, and it perturbs/shrinks the network's effective capacity. The
second is a generic regularisation-flavoured change, and on its own it would be an
uninteresting explanation of the gain.

This script separates the two by keeping the head in place and only disturbing it:

    intact      the untouched model (one reference per dataset)
    removed     the head's attention context zeroed — identical to the sweep
    shuffled    the head's own weight values randomly permuted among themselves:
                same values, same count, same magnitudes, same head, same location,
                only the arrangement is scrambled  (n trials)
    noise       Gaussian noise added to the head's parameters, per-tensor sigma equal to
                that tensor slice's own std — a capacity/regularisation perturbation of
                matched scale that preserves the arrangement  (n trials)

Both controls are run on the top-k heads whose removal *helped* and the top-k whose
removal *hurt*, which makes the demonstration two-sided: if arrangement is what matters,
scrambling it should track removal on helpful heads, while matched-scale noise (which
leaves the arrangement intact) should not.

Heads are selected per dataset from the sweep CSV, because "helpful to remove" is a
per-dataset property. A dataset with no head of a given sign contributes none of that
side.

Usage:
    python eval_control_clap.py                          # every dataset in the sweep
    python eval_control_clap.py --datasets KAUH ICBHI
    python eval_control_clap.py --top-k 3 --trials 10
    python eval_control_clap.py --resume                  # continue a timed-out job
"""

import argparse
import csv
import hashlib
import random
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from eval_knn_clap import ClapBackbone
from eval_prune_clap import cache_split, encode_cached, prune_units
from knn_eval_core import build_dataset, knn_evaluate

CSV_FIELDS = [
    "dataset",
    "seed",
    "pooling",
    "selection",
    "rank",
    "pruning_index",
    "pruning_id",
    "condition",
    "trial",
    "trial_seed",
    "sweep_effect",
    "AUROC",
]


# ------------------------------------------------------------------- head selection


def select_heads(sweep_csv, none_csv, top_k):
    """Per dataset: the top-k heads whose removal helped and the top-k whose removal hurt."""
    sweep = pd.read_csv(sweep_csv)
    baseline = pd.read_csv(none_csv).set_index("dataset")["AUROC"]
    sweep["effect"] = sweep["AUROC"] - sweep["dataset"].map(baseline)
    if sweep["effect"].isna().any():
        missing = sorted(sweep.loc[sweep["effect"].isna(), "dataset"].unique())
        raise SystemExit(f"No intact reference for {missing} in {none_csv}")

    picked = []
    for dataset, group in sweep.groupby("dataset"):
        # "Helpful" and "harmful" are defined by the sign, so a dataset with fewer than
        # top_k of a sign simply contributes fewer heads — never a head of the wrong sign.
        for selection, subset, ascending in (
            ("helpful", group[group["effect"] > 0], False),
            ("harmful", group[group["effect"] < 0], True),
        ):
            chosen = subset.sort_values("effect", ascending=ascending).head(top_k)
            for rank, (_, row) in enumerate(chosen.iterrows(), start=1):
                picked.append(
                    {
                        "dataset": dataset,
                        "selection": selection,
                        "rank": rank,
                        "pruning_index": int(row["pruning_index"]),
                        "pruning_id": row["pruning_id"],
                        "sweep_effect": float(row["effect"]),
                    }
                )
            if chosen.empty:
                print(f"  {dataset}: no {selection}-to-remove head; skipped")
    return pd.DataFrame(picked)


# ------------------------------------------------------------- head-local mutations


def head_parameters(model, pruning_id):
    """Every parameter that belongs to one head, as in-place views.

    A head owns its rows of Q/K/V (weights and biases), its columns of the attention
    output projection, and its column of the Swin relative-position-bias table. Basic
    slicing returns views, so writing through these writes into the live model.
    """
    stage, block, head = (int(value) for value in re.match(r"s(\d+)\.b(\d+)\.h(\d+)", pruning_id).groups())
    layer = model.audio_model.audio_encoder.layers[stage].blocks[block]
    attention = layer.attention.self
    size = attention.attention_head_size
    rows = slice(head * size, (head + 1) * size)

    views = []
    for projection in (attention.query, attention.key, attention.value):
        views.append(projection.weight.data[rows, :])
        if projection.bias is not None:
            views.append(projection.bias.data[rows])
    views.append(layer.attention.output.dense.weight.data[:, rows])
    views.append(attention.relative_position_bias_table.data[:, head])
    return views


@contextmanager
def mutated_head(model, pruning_id, mode, seed, noise_scale=1.0):
    """Shuffle or noise one head's parameters in place, then restore them exactly.

    Both mutations are generated on the CPU from an explicitly seeded generator, so a
    trial is reproducible and independent of device and of how many trials ran before it.
    """
    views = head_parameters(model, pruning_id)
    originals = [view.clone() for view in views]
    generator = torch.Generator().manual_seed(seed)

    try:
        for view in views:
            values = view.detach().to("cpu", copy=True)
            if mode == "shuffle":
                flat = values.reshape(-1)
                # Same values, same count, same magnitudes — only the arrangement moves.
                replacement = flat[torch.randperm(flat.numel(), generator=generator)].reshape(values.shape)
            elif mode == "noise":
                # Matched scale: sigma is this slice's own std (times --noise-scale), so
                # the perturbation is a capacity-style disturbance of comparable size with
                # the arrangement left intact.
                sigma = float(values.std()) * noise_scale
                replacement = values + torch.randn(values.shape, generator=generator) * sigma
            else:
                raise ValueError(f"Unknown mutation mode: {mode}")
            view.copy_(replacement.to(view.device))
        yield
    finally:
        for view, original in zip(views, originals):
            view.copy_(original)


@contextmanager
def unchanged():
    yield


MUTATION = {"shuffled": "shuffle", "noise": "noise"}


def trial_seed(seed, head_id, condition, trial):
    """Deterministic across processes — Python's hash() is salted per run, so never use it."""
    key = f"{seed}|{head_id}|{condition}|{trial}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % (2**31)


# ------------------------------------------------------------------------------ CLI


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-csv", default=None, help="Defaults to the head sweep of --pooling/--seed.")
    parser.add_argument("--none-csv", default=None, help="Defaults to the none run of --pooling/--seed.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--pooling", choices=list(ClapBackbone.pooling_choices), default="projected")
    parser.add_argument("--top-k", type=int, default=3, help="Heads per sign per dataset.")
    parser.add_argument("--trials", type=int, default=10, help="Random draws per control condition.")
    parser.add_argument(
        "--noise-scale", type=float, default=1.0,
        help="Multiplier on the noise sigma (1.0 = the parameters' own std). A run with a "
             "different scale writes its own CSV, since the scale is part of the condition.",
    )
    parser.add_argument("--config", default="label-mapping.yaml")
    parser.add_argument("--model", default=ClapBackbone.default_model_id)
    parser.add_argument("-k", "--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-seconds", type=int, default=ClapBackbone.default_clip_seconds)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trial_plan(trials):
    """(condition, trial) pairs per head: the deterministic removal, then the random draws."""
    plan = [("removed", 0)]
    plan += [("shuffled", trial) for trial in range(1, trials + 1)]
    plan += [("noise", trial) for trial in range(1, trials + 1)]
    return plan


def open_csv(path, resume):
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if resume and path.exists():
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                done.add((row["dataset"], int(row["pruning_index"]), row["condition"], int(row["trial"])))
        out = open(path, "a", newline="")
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
    else:
        out = open(path, "w", newline="")
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
        writer.writeheader()
    return out, writer, done


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    sweep_csv = Path(args.sweep_csv or f"results/prune_clap_head_{args.pooling}_seed{args.seed}.csv")
    none_csv = Path(args.none_csv or f"results/prune_clap_none_{args.pooling}_seed{args.seed}.csv")
    for path in (sweep_csv, none_csv):
        if not path.exists():
            raise SystemExit(f"Missing {path}; run eval_prune_clap.py first.")

    print(f"Selecting heads from {sweep_csv}")
    heads = select_heads(sweep_csv, none_csv, args.top_k)
    if args.datasets:
        heads = heads[heads["dataset"].isin(args.datasets)]
    if heads.empty:
        raise SystemExit("No heads selected.")
    print(
        f"{len(heads)} head(s) over {heads['dataset'].nunique()} dataset(s): "
        f"{(heads['selection'] == 'helpful').sum()} helpful, {(heads['selection'] == 'harmful').sum()} harmful"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} (audio encoder only, pooling={args.pooling})")
    backbone = ClapBackbone(args.model, args.pooling, device, args.clip_seconds)
    pruners = dict(prune_units(backbone.model, "head"))

    entries = {entry["name"]: entry for entry in yaml.safe_load(open(args.config))["datasets"]}

    scale_tag = "" if args.noise_scale == 1.0 else f"_noise{args.noise_scale:g}"
    output = Path(
        args.output or f"results/control_clap_head_{args.pooling}_seed{args.seed}{scale_tag}.csv"
    )
    handle, writer, done = open_csv(output, args.resume)
    if done:
        print(f"Resuming {output}: {len(done)} row(s) already present")

    plan = trial_plan(args.trials)

    try:
        for dataset in sorted(heads["dataset"].unique()):
            selected = heads[heads["dataset"] == dataset]
            entry = entries.get(dataset)
            if entry is None:
                print(f"\n=== {dataset} ===\n  skipped: not in {args.config}")
                continue
            print(f"\n=== {dataset} ===  {len(selected)} head(s)", flush=True)

            # The intact model is the same for every head, so it is one reference row
            # per dataset rather than one per head.
            reference_row = {
                "selection": "reference", "rank": 0, "pruning_index": -1,
                "pruning_id": "none", "sweep_effect": 0.0,
            }
            pending = []
            if (dataset, -1, "intact", 0) not in done:
                pending.append((reference_row, "intact", 0))
            pending += [
                (row, condition, trial)
                for _, row in selected.iterrows()
                for condition, trial in plan
                if (dataset, row["pruning_index"], condition, trial) not in done
            ]

            if not pending:
                print("  skipped: already complete in the resumed CSV")
                continue

            try:
                train_set = build_dataset(
                    entry["path"], entry["label_map"], "train", args.clip_seconds, backbone.sample_rate
                )
                test_set = build_dataset(
                    entry["path"], entry["label_map"], "test", args.clip_seconds, backbone.sample_rate
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"  skipped: {exc}")
                continue

            train_batches, train_y = cache_split(
                train_set, backbone, args.batch_size, args.num_workers, f"{dataset} train (cache)"
            )
            test_batches, test_y = cache_split(
                test_set, backbone, args.batch_size, args.num_workers, f"{dataset} test (cache)"
            )

            for row, condition, trial in pending:
                head_id = row["pruning_id"]
                # Distinct per (head, condition, trial) so no two draws share a stream.
                draw_seed = trial_seed(args.seed, head_id, condition, trial)

                if condition == "intact":
                    context = unchanged()
                elif condition == "removed":
                    context = pruners[head_id]()
                else:
                    context = mutated_head(
                        backbone.model, head_id, MUTATION[condition], draw_seed, args.noise_scale
                    )

                with context:
                    train_x = encode_cached(train_batches, backbone)
                    test_x = encode_cached(test_batches, backbone)

                auroc = knn_evaluate(train_x, train_y, test_x, test_y, args.k)["auroc"]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "seed": args.seed,
                        "pooling": args.pooling,
                        "selection": row["selection"],
                        "rank": row["rank"],
                        "pruning_index": row["pruning_index"],
                        "pruning_id": head_id,
                        "condition": condition,
                        "trial": trial,
                        "trial_seed": draw_seed if condition in MUTATION else "",
                        "sweep_effect": f"{row['sweep_effect']:.6f}",
                        "AUROC": f"{auroc:.6f}",
                    }
                )
                handle.flush()
                print(f"  {head_id:<12} {row['selection']:<8} {condition:<9} t{trial:<3} auroc={auroc:.4f}", flush=True)

            del train_batches, test_batches
    finally:
        handle.close()

    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
