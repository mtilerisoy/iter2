"""What is special about the heads whose removal helps? Measure properties, not locations.

The sweep says *which* heads matter and the shuffle control says the effect is about the
head's specific arrangement of weights. This script asks *why those heads*: it measures
standard, cheap properties of all 184 heads and leaves it to ``analyze_head_properties.py``
to ask which properties separate helpful-to-remove from harmful-to-remove.

Two families of property:

**Static (weights only, no data).** Norms of the head's Q/K/V rows and output-projection
columns; how concentrated those weights are (excess kurtosis, participation ratio,
max/RMS); and the spectral norm of the head's OV circuit ``W_out[:, head] @ W_v[head, :]``,
i.e. how hard the head can write into the residual stream regardless of input.

**Activation (one forward pass per corpus).** For every head, how much it actually writes
when audio goes through:

    ctx_rms       RMS L2 norm of the head's attention context, per token
    contrib_rms   RMS L2 norm of what the head adds to the residual stream — the context
                  passed through that head's slice of the output projection
    energy_share  contrib energy as a fraction of all heads in the same block

Every head is instrumented at once, so a corpus costs **one** pass, not 184.

The general-audio corpus (FSD50k) is the point of the exercise: CLAP was pretrained on
general audio, so a head that writes hard on general audio but whose removal *helps* the
medical probe is doing a job that conflicts with the medical task.

Usage:
    python eval_head_properties.py
    python eval_head_properties.py --datasets KAUH ICBHI --general-limit 200
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import AudioDataset
from eval_knn_clap import ClapBackbone
from knn_eval_core import build_dataset


# ------------------------------------------------------------------ static properties


def head_slices(model):
    """(head_id, stage, block, head, attention module, block module) for all 184 heads."""
    for stage_idx, stage in enumerate(model.audio_model.audio_encoder.layers):
        for block_idx, block in enumerate(stage.blocks):
            attention = block.attention.self
            for head in range(attention.num_attention_heads):
                yield f"s{stage_idx}.b{block_idx}.h{head}", stage_idx, block_idx, head, attention, block


@torch.no_grad()
def static_properties(model):
    """Weight-only descriptors of every head."""
    rows = []
    for head_id, stage, block_idx, head, attention, block in head_slices(model):
        size = attention.attention_head_size
        rows_slice = slice(head * size, (head + 1) * size)

        qkv = torch.cat(
            [
                projection.weight.data[rows_slice, :].reshape(-1)
                for projection in (attention.query, attention.key, attention.value)
            ]
        )
        w_out = block.attention.output.dense.weight.data[:, rows_slice]
        w_value = attention.value.weight.data[rows_slice, :]
        every = torch.cat([qkv, w_out.reshape(-1)])

        # Concentration: is the head's mass spread over its weights or held by a few?
        centred = every - every.mean()
        variance = float((centred**2).mean())
        kurtosis = float((centred**4).mean() / variance**2 - 3.0) if variance > 0 else float("nan")
        squares = every**2
        participation = float(squares.sum() ** 2 / (squares**2).sum() / every.numel())

        # OV circuit: what this head can write into the residual stream, input aside.
        ov_spectral = float(torch.linalg.matrix_norm(w_out @ w_value, ord=2))

        rows.append(
            {
                "pruning_id": head_id,
                "stage": stage,
                "block": block_idx,
                "head": head,
                "head_size": size,
                "n_params": int(every.numel()),
                "w_norm": float(every.norm()),
                "w_norm_qkv": float(qkv.norm()),
                "w_norm_out": float(w_out.norm()),
                "w_rms": float(every.pow(2).mean().sqrt()),
                "w_kurtosis": kurtosis,
                "w_participation": participation,
                "w_max_over_rms": float(every.abs().max() / every.pow(2).mean().sqrt()),
                "ov_spectral": ov_spectral,
            }
        )
    return pd.DataFrame(rows)


# -------------------------------------------------------------- activation properties


class HeadActivationProbe:
    """Accumulates per-head activation energy over a corpus, all heads in one pass.

    The energy a head writes into the residual stream is computed without materialising
    that contribution: for context vectors ``c`` and the head's output-projection slice
    ``W``, ``sum_t ||W c_t||^2 == sum(  (W^T W) * sum_t c_t c_t^T  )``, and both factors
    are only head_size x head_size.
    """

    def __init__(self, model):
        self.totals = {
            head_id: {"ctx_sumsq": 0.0, "contrib_sumsq": 0.0, "vectors": 0}
            for head_id, *_ in head_slices(model)
        }
        # One hook per block, not per head: a block's hook records all of its heads.
        self.handles = [
            block.attention.self.register_forward_hook(
                self._make_hook(stage_idx, block_idx, block.attention.self, block)
            )
            for stage_idx, stage in enumerate(model.audio_model.audio_encoder.layers)
            for block_idx, block in enumerate(stage.blocks)
        ]

    def _make_hook(self, stage_idx, block_idx, attention, block):
        heads = attention.num_attention_heads
        size = attention.attention_head_size
        w_out = block.attention.output.dense.weight  # [dim, dim], head h owns its columns

        def hook(module, args, output):
            context = output[0].detach()
            windows, tokens, _ = context.shape
            reshaped = context.view(windows, tokens, heads, size)

            # gram[h] = sum over all (window, token) of c c^T for head h.
            gram = torch.einsum("nthd,nthe->hde", reshaped, reshaped).float()
            ctx_sumsq = gram.diagonal(dim1=1, dim2=2).sum(-1)

            slices = w_out[:, : heads * size].view(-1, heads, size)
            metric = torch.einsum("nhd,nhe->hde", slices, slices).float()
            contrib_sumsq = (metric * gram).sum(dim=(1, 2))

            for head in range(heads):
                totals = self.totals[f"s{stage_idx}.b{block_idx}.h{head}"]
                totals["ctx_sumsq"] += float(ctx_sumsq[head])
                totals["contrib_sumsq"] += float(contrib_sumsq[head])
                totals["vectors"] += windows * tokens

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def frame(self, corpus):
        rows = []
        for head_id, totals in self.totals.items():
            count = max(totals["vectors"], 1)
            rows.append(
                {
                    "pruning_id": head_id,
                    "corpus": corpus,
                    "ctx_rms": float(np.sqrt(totals["ctx_sumsq"] / count)),
                    "contrib_rms": float(np.sqrt(totals["contrib_sumsq"] / count)),
                    "contrib_sumsq": totals["contrib_sumsq"],
                    "n_vectors": count,
                }
            )
        frame = pd.DataFrame(rows)
        frame["block_id"] = frame["pruning_id"].str.rsplit(".", n=1).str[0]
        # Share of its own block's written energy: comparable across stages, which differ
        # in width, head count and overall activation scale.
        frame["energy_share"] = frame["contrib_sumsq"] / frame.groupby("block_id")[
            "contrib_sumsq"
        ].transform("sum")
        return frame.drop(columns=["block_id"])


@torch.no_grad()
def probe_corpus(dataset, backbone, batch_size, num_workers, corpus, limit=None):
    """One instrumented pass over a corpus; returns per-head activation statistics."""
    if limit is not None and limit < len(dataset):
        # Deterministic subsample so a corpus costs the same regardless of its size.
        indices = np.random.default_rng(0).choice(len(dataset), size=limit, replace=False)
        dataset = torch.utils.data.Subset(dataset, sorted(indices.tolist()))
        collate = dataset.dataset.collate_fn
    else:
        collate = dataset.collate_fn

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate,
    )

    probe = HeadActivationProbe(backbone.model)
    try:
        for waveforms, _ in tqdm(loader, desc=f"{corpus} probe", leave=False):
            inputs = backbone.prepare([w.numpy() for w in waveforms])
            inputs = {k: v.to(backbone.device) for k, v in inputs.items()}
            backbone.encode(inputs)
    finally:
        probe.remove()

    frame = probe.frame(corpus)
    frame["n_recordings"] = len(dataset)
    return frame


def general_dataset(config_path, clip_seconds, sample_rate, use_all_labels):
    """FSD50k as a *probe corpus*, not a task: labels are irrelevant to activation stats."""
    entry = yaml.safe_load(open(config_path))["datasets"][0]
    manifest = pd.read_json(entry["path"], lines=True)
    if use_all_labels:
        # The config's label_map covers 2 of ~200 classes because it defines a binary
        # task. For a general-audio probe we want the breadth, so every label is kept.
        label_map = {label: 0 for label in manifest["label"].unique()}
    else:
        label_map = entry["label_map"]

    split = "test" if (manifest["split"] == "test").any() else "train"
    return build_dataset(entry["path"], label_map, split, clip_seconds, sample_rate), entry["name"]


# ------------------------------------------------------------------------------- run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="label-mapping.yaml")
    parser.add_argument("--general-config", default="label-mapping-FSD50k.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--model", default=ClapBackbone.default_model_id)
    parser.add_argument("--pooling", choices=list(ClapBackbone.pooling_choices), default="projected")
    parser.add_argument("--split", default="test", choices=("train", "test"),
                        help="Medical split to probe; test is where the sweep effect was measured.")
    parser.add_argument("--general-limit", type=int, default=1000,
                        help="Recordings sampled from FSD50k (0 = all).")
    parser.add_argument(
        "--general-labels", choices=("all", "mapped"), default="all",
        help="'all' uses every FSD50k class as a general-audio probe; 'mapped' restricts "
             "to the two classes the config's binary task defines.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-seconds", type=int, default=ClapBackbone.default_clip_seconds)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} (pooling={args.pooling})")
    backbone = ClapBackbone(args.model, args.pooling, device, args.clip_seconds)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    static = static_properties(backbone.model)
    static_path = out_dir / f"head_static_{args.pooling}_seed{args.seed}.csv"
    static.to_csv(static_path, index=False, float_format="%.6g")
    print(f"Wrote {static_path}  ({len(static)} heads)")

    entries = yaml.safe_load(open(args.config))["datasets"]
    if args.datasets:
        entries = [e for e in entries if e["name"] in args.datasets]

    frames = []
    for entry in entries:
        print(f"\n=== {entry['name']} ({args.split}) ===", flush=True)
        try:
            dataset = build_dataset(
                entry["path"], entry["label_map"], args.split, args.clip_seconds, backbone.sample_rate
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"  skipped: {exc}")
            continue
        if len(dataset) == 0:
            print("  skipped: empty split")
            continue
        print(f"  {len(dataset)} recordings")
        frame = probe_corpus(dataset, backbone, args.batch_size, args.num_workers, entry["name"])
        frame["domain"] = "medical"
        frames.append(frame)

    if Path(args.general_config).exists():
        general, name = general_dataset(
            args.general_config, args.clip_seconds, backbone.sample_rate, args.general_labels == "all"
        )
        limit = args.general_limit or None
        print(f"\n=== {name} (general audio) ===\n  {len(general)} recordings, limit={limit}", flush=True)
        frame = probe_corpus(
            general, backbone, args.batch_size, args.num_workers, name, limit=limit
        )
        frame["domain"] = "general"
        frames.append(frame)
    else:
        print(f"\nNo {args.general_config}; skipping the general-audio probe.")

    activations = pd.concat(frames, ignore_index=True)
    activation_path = out_dir / f"head_activations_{args.pooling}_seed{args.seed}.csv"
    activations.to_csv(activation_path, index=False, float_format="%.6g")
    print(f"\nWrote {activation_path}  ({activations['corpus'].nunique()} corpora)")


if __name__ == "__main__":
    main()
