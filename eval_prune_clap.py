"""Structured pruning evaluation of CLAP audio embeddings on the binary medical benchmarks.

Entire attention heads or entire transformer blocks are zeroed from the audio tower of
CLAP, then the crippled model is used to extract one embedding per recording. A k-NN
classifier (k=5, cosine) is fitted on the train split and evaluated on the test split of
each dataset in ``label-mapping.yaml``, exactly as in ``eval_knn_clap.py``.

The sweep visits every prunable unit one at a time (single-unit ablation, never
cumulative): 184 heads or 12 blocks for ``laion/clap-htsat-unfused``. ``--prune_type
none`` evaluates the intact model once, which is the reference row for every sweep.

Pruning is applied as *masking*, not surgery: a head is removed by zeroing its slice of
the self-attention context (identical to deleting its rows from the output projection),
and a block is removed by replacing it with the identity (its residual stream passes
through untouched). Weights are never modified, so the same in-memory model serves the
whole sweep.

Because the sweep re-embeds the same audio hundreds of times, decoded + feature-extracted
inputs are cached in RAM once per dataset; only the GPU forward pass is repeated.

Nothing here is stochastic, but the seed is still set (and recorded per row) so the CSV
is self-describing and a future stochastic variant stays comparable.

Usage:
    python eval_prune_clap.py --prune_type none
    python eval_prune_clap.py --prune_type block
    python eval_prune_clap.py --prune_type head --datasets KAUH ICBHI
    python eval_prune_clap.py --prune_type head --pooling pooled
"""

import argparse
import csv
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_knn_clap import ClapBackbone
from knn_eval_core import build_dataset, knn_evaluate

CSV_FIELDS = [
    "dataset",
    "seed",
    "pruning_type",
    "pruning_index",
    "pruning_id",
    "pooling",
    "AUROC",
]


# --------------------------------------------------------------------------- pruning


def audio_blocks(model):
    """Every transformer block of the HTSAT audio tower, as (stage_idx, block_idx, block)."""
    return [
        (stage_idx, block_idx, block)
        for stage_idx, stage in enumerate(model.audio_model.audio_encoder.layers)
        for block_idx, block in enumerate(stage.blocks)
    ]


def prune_units(model, prune_type):
    """The ordered list of prunable units; ``pruning_index`` indexes into it.

    Each entry is ``(id_string, apply_fn)`` where ``apply_fn(model)`` is a context
    manager that removes that one unit for the duration of the ``with`` block.
    """
    if prune_type == "none":
        return [("none", _prune_nothing)]

    units = []
    for stage_idx, block_idx, block in audio_blocks(model):
        if prune_type == "block":
            units.append(
                (f"s{stage_idx}.b{block_idx}", _make_block_pruner(block))
            )
        else:
            for head in range(block.attention.self.num_attention_heads):
                units.append(
                    (f"s{stage_idx}.b{block_idx}.h{head}", _make_head_pruner(block, head))
                )
    return units


@contextmanager
def _prune_nothing():
    yield


def _make_head_pruner(block, head):
    """Zero one head's slice of the attention context (== deleting the head)."""
    attention = block.attention.self
    head_size = attention.attention_head_size
    start, end = head * head_size, (head + 1) * head_size

    def hook(module, args, output):
        context = output[0].clone()
        context[..., start:end] = 0.0
        return (context,) + tuple(output[1:])

    @contextmanager
    def pruner():
        handle = attention.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    return pruner


def _make_block_pruner(block):
    """Replace one block with the identity, leaving the residual stream untouched."""

    def identity_forward(
        hidden_states,
        input_dimensions,
        head_mask=None,
        output_attentions=False,
        always_partition=False,
    ):
        return (hidden_states,)

    @contextmanager
    def pruner():
        original = block.forward
        block.forward = identity_forward
        try:
            yield
        finally:
            block.forward = original

    return pruner


# ------------------------------------------------------------------- feature caching


def cache_split(dataset, backbone, batch_size, num_workers, desc):
    """Decode + feature-extract a split once; the sweep then only repeats the forward pass."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
    )

    batches, labels = [], []
    for waveforms, batch_labels in tqdm(loader, desc=desc, leave=False):
        inputs = backbone.prepare([w.numpy() for w in waveforms])
        batches.append({k: v for k, v in inputs.items()})
        labels.append(batch_labels.numpy())

    return batches, np.concatenate(labels)


@torch.no_grad()
def encode_cached(batches, backbone):
    feats = [
        backbone.encode({k: v.to(backbone.device) for k, v in batch.items()})
        .float()
        .cpu()
        .numpy()
        for batch in batches
    ]
    return np.concatenate(feats)


# ------------------------------------------------------------------------------ CLI


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prune_type",
        choices=("head", "block", "none"),
        default="head",
        help="Sweep over attention heads, over transformer blocks, or evaluate intact.",
    )
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of config names.")
    parser.add_argument(
        "--pooling",
        choices=list(ClapBackbone.pooling_choices),
        default="projected",
        help="'projected': 512-d shared-space vector. 'pooled': 768-d pre-projection.",
    )
    parser.add_argument("--config", default="label-mapping.yaml")
    parser.add_argument("--model", default=ClapBackbone.default_model_id)
    parser.add_argument("-k", "--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-direction", choices=("forward", "reverse"), default="forward",
        help="'forward' fits the k-NN on train and scores test (the standard protocol). "
             "'reverse' swaps the two roles, using the same two groups of recordings, so "
             "the AUROC is measured on recordings the forward run never scored. Nothing "
             "else changes and no new split is invented.",
    )
    parser.add_argument("--clip-seconds", type=int, default=ClapBackbone.default_clip_seconds)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=None, help="Defaults to results/prune_clap_<...>.csv")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing CSV, skipping (dataset, pruning_index) pairs already in it.",
    )
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_entries(config_path, wanted):
    entries = yaml.safe_load(open(config_path))["datasets"]
    if wanted:
        wanted = set(wanted)
        entries = [e for e in entries if e["name"] in wanted]
        missing = wanted - {e["name"] for e in entries}
        if missing:
            raise SystemExit(f"Unknown dataset(s) in --datasets: {sorted(missing)}")
    return entries


def open_csv(path, resume):
    """Return (file handle, writer, already-done set). Rows are flushed as they land."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if resume and path.exists():
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                done.add((row["dataset"], int(row["pruning_index"])))
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} (audio encoder only, pooling={args.pooling})")
    backbone = ClapBackbone(args.model, args.pooling, device, args.clip_seconds)
    print(f"Audio format: {backbone.sample_rate} Hz, {args.clip_seconds}s clips")

    units = prune_units(backbone.model, args.prune_type)
    print(f"Pruning sweep: {args.prune_type} ({len(units)} configuration(s))")
    print(f"Split direction: {args.split_direction}"
          + ("  (k-NN fitted on test, scored on train)" if args.split_direction == "reverse" else ""))

    entries = load_entries(args.config, args.datasets)

    direction_tag = "" if args.split_direction == "forward" else "_reverse"
    output = Path(
        args.output
        or f"results/prune_clap_{args.prune_type}_{args.pooling}_seed{args.seed}{direction_tag}.csv"
    )
    handle, writer, done = open_csv(output, args.resume)
    if done:
        print(f"Resuming {output}: {len(done)} row(s) already present")

    try:
        for entry in entries:
            name, jsonl_path, label_map = entry["name"], entry["path"], entry["label_map"]
            print(f"\n=== {name} ===", flush=True)

            try:
                train_set = build_dataset(
                    jsonl_path, label_map, "train", args.clip_seconds, backbone.sample_rate
                )
                test_set = build_dataset(
                    jsonl_path, label_map, "test", args.clip_seconds, backbone.sample_rate
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"  skipped: {exc}")
                continue

            dropped = train_set.num_dropped + test_set.num_dropped
            print(f"  train={len(train_set)} test={len(test_set)} (dropped {dropped} unmapped)")
            if len(train_set) == 0 or len(test_set) == 0:
                print("  skipped: empty split")
                continue

            pending = [i for i in range(len(units)) if (name, i) not in done]
            if not pending:
                print("  skipped: already complete in the resumed CSV")
                continue

            # One decode + feature-extraction pass, reused by every pruning configuration.
            train_batches, train_y = cache_split(
                train_set, backbone, args.batch_size, args.num_workers, f"{name} train (cache)"
            )
            test_batches, test_y = cache_split(
                test_set, backbone, args.batch_size, args.num_workers, f"{name} test (cache)"
            )

            for index in tqdm(pending, desc=f"{name} {args.prune_type}s"):
                unit_id, pruner = units[index]
                with pruner():
                    train_x = encode_cached(train_batches, backbone)
                    test_x = encode_cached(test_batches, backbone)

                # The reverse direction reuses the very same embeddings and only swaps
                # which split fits the classifier and which one is scored.
                if args.split_direction == "forward":
                    fit_x, fit_y, score_x, score_y = train_x, train_y, test_x, test_y
                else:
                    fit_x, fit_y, score_x, score_y = test_x, test_y, train_x, train_y
                auroc = knn_evaluate(fit_x, fit_y, score_x, score_y, args.k)["auroc"]
                writer.writerow(
                    {
                        "dataset": name,
                        "seed": args.seed,
                        "pruning_type": args.prune_type,
                        "pruning_index": index if args.prune_type != "none" else -1,
                        "pruning_id": unit_id,
                        "pooling": args.pooling,
                        "AUROC": f"{auroc:.6f}",
                    }
                )
                handle.flush()
                tqdm.write(f"  [{index:>3}] {unit_id:<14} auroc={auroc:.4f}")

            del train_batches, test_batches
    finally:
        handle.close()

    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
