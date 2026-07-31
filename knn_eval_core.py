"""Shared machinery for k-NN probing of frozen audio backbones.

Each backbone turns a batch of raw waveforms into one embedding per recording; this
module owns everything else — split construction from ``label-mapping.yaml``, the
embedding loop, the k-NN classifier and the metric/report plumbing — so that adding a
model means implementing :class:`Backbone` and nothing more.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import AudioDataset


class Backbone:
    """A frozen audio encoder used as a fixed feature extractor.

    Subclasses declare the audio format they need and how to pool a batch into
    embeddings. The text tower of a multimodal model (if any) is never loaded.
    """

    name = "backbone"
    default_model_id = None
    sample_rate = 16000
    default_clip_seconds = 10
    pooling_choices = ("default",)
    default_pooling = "default"

    def __init__(self, model_id, pooling, device, clip_seconds):
        self.model_id = model_id
        self.pooling = pooling
        self.device = device
        self.clip_seconds = clip_seconds

    def prepare(self, waveforms):
        """Feature-extract a list of 1-D float32 numpy waveforms into model inputs."""
        raise NotImplementedError

    def encode(self, inputs):
        """Return a [batch, dim] float tensor of embeddings."""
        raise NotImplementedError


def build_dataset(jsonl_path, label_map, split, clip_seconds, sample_rate):
    """Deterministic, un-augmented view of one split at the backbone's sample rate."""
    return AudioDataset(
        data_path=jsonl_path,
        label_map=label_map,
        target_audio_seconds=clip_seconds,
        mode=split,
        apply_augmentation=False,
        return_melspectrogram=False,
        sample_rate=sample_rate,
        random_crop=False,
        drop_unmapped=True,
    )


@torch.no_grad()
def embed_split(dataset, backbone, batch_size, num_workers, desc):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    embeddings, labels = [], []
    for waveforms, batch_labels in tqdm(loader, desc=desc, leave=False):
        inputs = backbone.prepare([w.numpy() for w in waveforms])
        inputs = {k: v.to(backbone.device) for k, v in inputs.items()}
        feats = backbone.encode(inputs)
        embeddings.append(feats.float().cpu().numpy())
        labels.append(batch_labels.numpy())

    return np.concatenate(embeddings), np.concatenate(labels)


def knn_evaluate(train_x, train_y, test_x, test_y, k):
    # Cosine similarity == inner product on unit-norm vectors; normalising here keeps
    # the neighbour search consistent regardless of embedding magnitude.
    train_x = train_x / np.clip(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12, None)
    test_x = test_x / np.clip(np.linalg.norm(test_x, axis=1, keepdims=True), 1e-12, None)

    effective_k = min(k, len(train_x))
    knn = KNeighborsClassifier(n_neighbors=effective_k, metric="cosine", weights="distance")
    knn.fit(train_x, train_y)

    pred = knn.predict(test_x)
    proba = knn.predict_proba(test_x)
    # Column index of the positive class; a single-class train split has no column for it.
    pos_idx = list(knn.classes_).index(1) if 1 in knn.classes_ else None
    scores = proba[:, pos_idx] if pos_idx is not None else np.zeros(len(test_x))

    tn, fp, fn, tp = confusion_matrix(test_y, pred, labels=[0, 1]).ravel()
    metrics = {
        "k": effective_k,
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "test_positive_rate": float(np.mean(test_y == 1)),
        "accuracy": float(accuracy_score(test_y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
        "f1_macro": float(f1_score(test_y, pred, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(test_y, pred, pos_label=1, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
    }
    metrics["auroc"] = (
        float(roc_auc_score(test_y, scores)) if len(np.unique(test_y)) == 2 else float("nan")
    )
    return metrics


def build_arg_parser(backbone_cls, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="label-mapping.yaml")
    parser.add_argument("--model", default=backbone_cls.default_model_id)
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of config names.")
    parser.add_argument("-k", "--k", type=int, default=5)
    parser.add_argument(
        "--pooling",
        choices=list(backbone_cls.pooling_choices),
        default=backbone_cls.default_pooling,
    )
    parser.add_argument("--clip-seconds", type=int, default=backbone_cls.default_clip_seconds)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=f"results/knn_{backbone_cls.name}.json")
    return parser


def run(backbone_cls, description):
    args = build_arg_parser(backbone_cls, description).parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} (audio encoder only, pooling={args.pooling})")
    backbone = backbone_cls(args.model, args.pooling, device, args.clip_seconds)
    print(f"Audio format: {backbone.sample_rate} Hz, {args.clip_seconds}s clips")

    config = yaml.safe_load(open(args.config))
    entries = config["datasets"]
    if args.datasets:
        wanted = set(args.datasets)
        entries = [e for e in entries if e["name"] in wanted]
        missing = wanted - {e["name"] for e in entries}
        if missing:
            raise SystemExit(f"Unknown dataset(s) in --datasets: {sorted(missing)}")

    results = {}
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

        train_x, train_y = embed_split(
            train_set, backbone, args.batch_size, args.num_workers, f"{name} train"
        )
        test_x, test_y = embed_split(
            test_set, backbone, args.batch_size, args.num_workers, f"{name} test"
        )

        metrics = knn_evaluate(train_x, train_y, test_x, test_y, args.k)
        metrics["embedding_dim"] = int(train_x.shape[1])
        results[name] = metrics
        print(
            f"  acc={metrics['accuracy']:.4f}  bal_acc={metrics['balanced_accuracy']:.4f}  "
            f"f1_macro={metrics['f1_macro']:.4f}  auroc={metrics['auroc']:.4f}",
            flush=True,
        )

    if not results:
        print("\nNo datasets evaluated.")
        return

    header = f"{'dataset':<12}{'acc':>8}{'bal_acc':>10}{'f1_macro':>10}{'auroc':>8}{'sens':>8}{'spec':>8}"
    print(f"\n{header}")
    for name, m in results.items():
        print(
            f"{name:<12}{m['accuracy']:>8.4f}{m['balanced_accuracy']:>10.4f}"
            f"{m['f1_macro']:>10.4f}{m['auroc']:>8.4f}{m['sensitivity']:>8.4f}{m['specificity']:>8.4f}"
        )
    means = {key: float(np.nanmean([m[key] for m in results.values()]))
             for key in ("accuracy", "balanced_accuracy", "f1_macro", "auroc")}
    print(f"{'MEAN':<12}{means['accuracy']:>8.4f}{means['balanced_accuracy']:>10.4f}"
          f"{means['f1_macro']:>10.4f}{means['auroc']:>8.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"backbone": backbone_cls.name, "model": args.model, "k": args.k,
         "pooling": args.pooling, "clip_seconds": args.clip_seconds,
         "sample_rate": backbone.sample_rate, "per_dataset": results, "mean": means},
        indent=2,
    ))
    print(f"\nWrote {out_path}")
