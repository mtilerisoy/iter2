# med-audio-base

Benchmarking frozen audio foundation models on **binary medical audio classification** —
lung sounds, heart sounds, and coughs — across 10 public datasets under one common
protocol.

The current experiment is **k-NN probing**: freeze a pretrained audio encoder, embed
every recording once, fit a k-NN classifier (k=5, cosine) on the train split, evaluate on
the test split. No fine-tuning, no gradient steps. This measures *how linearly separable
the pathology signal already is* in each backbone's representation space, which is a
cheap and honest baseline before committing to fine-tuning experiments.

---

## Layout

```
label-mapping.yaml        Single source of truth: which datasets exist, where their
                          manifests live, and how raw labels collapse to binary 0/1.
dataloader.py             AudioDataset — manifest -> (waveform, binary label) tensors.
knn_eval_core.py          Everything shared by every k-NN experiment: split building,
                          embedding loop, k-NN, metrics, CLI, reporting.
eval_knn_clap.py          Thin backbone definitions. One file per model, ~40 lines each.
eval_knn_ast.py
eval_knn_whisper.py
scripts/knn-*.sh          Slurm batch scripts, one per backbone.
dataset/*.jsonl           Per-dataset manifests (the data contract, see below).
dataset/preprocess_*.py   One-off scripts that built those manifests from raw archives.
dataset/stratify.py       Helper to add stratified train/test splits to a manifest.
results/knn_*.json        Machine-readable results, one file per backbone.
```

Audio itself is **not** in this repo. Manifests point at absolute paths under
`/projects/prjs1635/datasets/`.

---

## The data contract

Every dataset is a JSONL manifest, one recording per line. Only four fields are load
bearing; the rest are dataset-specific extras that the loader ignores.

| field | required | meaning |
|---|---|---|
| `audio_file` | **yes** | Absolute path to a readable audio file. |
| `label` | **yes** | Raw dataset-native label, `str` or `int`. |
| `split` | **yes** | `"train"` or `"test"`. |
| `identifier` | no | Stable per-recording ID. |
| `duration` | no | Seconds. Informational only — not used for batching. |

If you add a dataset, produce a manifest with those fields and register it in
`label-mapping.yaml`. Nothing else needs to change.

> **Caveat:** `hflung_cycle.jsonl` has no `split` field and is not registered in the
> config. `fsd50k.jsonl` is registered but commented out (it is a general-audio sanity
> check, not a medical task). Both will fail if you naively point the loader at them.

### `label-mapping.yaml`

```yaml
- name: CirCor
  path: dataset/circor.jsonl
  label_map:
    Absent: 0
    Present: 1
    # Unknown is intentionally excluded
```

`label_map` maps raw labels to `{0 = healthy/negative, 1 = pathological/positive}`.
Rows whose label is absent from the map are **dropped** (`drop_unmapped=True`), and the
count is printed per dataset. This is deliberate: CirCor's `Unknown` is a genuine
"could not determine" class, not a third category to predict.

Three type gotchas that have already bitten this repo:

1. **Keys are parsed as YAML, so types must match the JSONL exactly.** BMD and CinC use
   integer labels (`0`/`1`, `-1`/`1`), and their YAML keys stay integers. Quote them and
   the lookup silently stops matching.
2. **HF_Lung's `nan: 0` matches the literal string `"nan"`** stored in the manifest — not
   a missing value. PyYAML only resolves `.nan`/`.NaN` (with a leading dot) to a float,
   so bare `nan` stays a string, which is what the data contains. If you ever regenerate
   that manifest with a real JSON `null`, pandas produces a float `NaN`, the key stops
   matching, and 6066 rows vanish silently.
3. **Verify coverage after editing.** An unmapped label is dropped, not flagged loudly.
   Compare the printed `train=/test=` counts against the manifest.

---

## `AudioDataset`

`dataloader.py` turns a manifest into `(waveform, label)` pairs. The parameters that
matter for evaluation:

| param | eval value | why |
|---|---|---|
| `sample_rate` | per-backbone | Each checkpoint has a required rate (CLAP 48 kHz, AST/Whisper 16 kHz). |
| `target_audio_seconds` | per-backbone | Fixed-length crop; shorter audio is zero-padded on the right. |
| `random_crop` | `False` | Deterministic crop from `t=0`, so embeddings are reproducible. Defaults to `mode == "train"`. |
| `apply_augmentation` | `False` | Augmentation is train-only by construction; never augment a probe. |
| `drop_unmapped` | `True` | See above. |
| `return_melspectrogram` | `False` | Backbones do their own feature extraction from raw waveform. |

`random_crop` is decoupled from `mode` on purpose: the k-NN probe reads the *train* split
but must embed it deterministically, which the old `mode == "train"` coupling made
impossible.

**Memory note.** `_read_audio` decodes each file in full before cropping, so peak RSS
scales with the longest recording, not the crop length. At 48 kHz this is what pushes
CLAP to ~10 GB — request `--mem=32G`. Passing `duration=` to `librosa.load` would cut
this substantially, but was deliberately not adopted: truncating before resampling is
not bit-identical to resampling then slicing, and it perturbed results by ~0.001–0.015
per metric. Exact reproducibility was judged worth the memory.

---

## Adding a new backbone

Subclass `Backbone`, declare the audio format, implement two methods. `knn_eval_core.run`
supplies the CLI, dataset iteration, metrics, and reporting.

```python
from knn_eval_core import Backbone, run

class MyBackbone(Backbone):
    name = "mymodel"                       # names results/knn_mymodel.json
    default_model_id = "org/checkpoint"
    sample_rate = 16000
    default_clip_seconds = 10
    pooling_choices = ("mean", "cls")      # exposed as --pooling
    default_pooling = "mean"

    def __init__(self, model_id, pooling, device, clip_seconds):
        super().__init__(model_id, pooling, device, clip_seconds)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    def prepare(self, waveforms):
        # list of 1-D float32 numpy arrays -> dict of CPU tensors
        return self.feature_extractor(waveforms, sampling_rate=self.sample_rate,
                                      return_tensors="pt")

    def encode(self, inputs):
        # dict of device tensors -> [batch, dim] embeddings
        return self.model(**inputs).last_hidden_state.mean(dim=1)

if __name__ == "__main__":
    run(MyBackbone, __doc__)
```

Then copy a `scripts/knn-*.sh`, change the job name and the python entrypoint.

Three things worth getting right, learned from the existing three:

- **Use only the audio tower.** CLAP's text encoder is never loaded; Whisper's decoder is
  loaded but never moved to the device. For classifier checkpoints, load the base model
  (`ASTModel`, not `ASTForAudioClassification`) so the pretraining head is dropped.
- **Respect the checkpoint's native sample rate.** Feeding 16 kHz audio to a 48 kHz model
  silently halves its effective bandwidth.
- **Watch fixed-size input windows.** Whisper always pads to 30 s regardless of input, so
  a 10 s clip occupies a third of the encoder output. `mean_valid` pooling averages only
  the real frames; averaging all 1500 dilutes every embedding with silence. Whisper's
  encoder has no attention mask, so padding still leaks in via self-attention — pooling
  limits the damage rather than eliminating it.

---

## Running

**Always via Slurm — never on the login node.** The `.out` logs are the record of what
was run.

```bash
sbatch scripts/knn-AST.sh
```

Each script requests 1×H100, 8 CPUs, 32 GB, 20 min, and writes `%j-%x.out` (jobid-jobname)
to the repo root. Inspect progress and results from that log:

```bash
grep -aE "^dataset|^MEAN|Traceback" 25064311-kNN-eval-AST.out
```

Useful flags when iterating (pass them inside the batch script):

```
--datasets KAUH ICBHI     # subset, for a fast smoke run
--pooling pooled          # per-backbone alternatives
-k 10                     # neighbourhood size
--clip-seconds 30         # longer context
--batch-size / --num-workers
```

Check peak memory of a finished job before scaling up:

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed
```

Note that `sacct` samples RSS periodically and under-reports transient spikes — an
8 GB job reporting 3 GB may still be one CoughVID batch away from an OOM kill.

---

## Results format

`results/knn_<backbone>.json` records the full configuration alongside the numbers, so a
result is interpretable without the log:

```json
{"backbone": "ast", "model": "MIT/ast-...", "k": 5, "pooling": "pooler",
 "clip_seconds": 10, "sample_rate": 16000,
 "per_dataset": {"KAUH": {"accuracy": ..., "balanced_accuracy": ..., "auroc": ...,
                          "sensitivity": ..., "specificity": ..., "f1_macro": ...,
                          "f1_positive": ..., "n_train": 225, "n_test": 111,
                          "test_positive_rate": 0.676, "embedding_dim": 768}},
 "mean": {...}}
```

### Current baselines (k=5, cosine, mean over 10 datasets)

| model | acc | bal_acc | f1_macro | auroc | dim | runtime | peak RSS |
|---|---|---|---|---|---|---|---|
| **AST** (AudioSet) | **0.8204** | **0.6810** | **0.6959** | **0.7778** | 768 | 5:47 | 4.9 GB |
| CLAP (htsat-unfused) | 0.7915 | 0.6305 | 0.6376 | 0.7174 | 512 | 8:14 | 10.2 GB |
| Whisper (base, encoder) | 0.7855 | 0.6192 | 0.6265 | 0.7078 | 512 | 7:57 | 5.2 GB |

AST wins on 9 of 10 datasets. The ordering tracks pretraining objective: AudioSet
supervision (which includes cough, breathing, and heartbeat classes) transfers best,
contrastive audio-text next, speech transcription worst — Whisper is trained to discard
exactly the non-speech texture these tasks depend on.

---

## Final Remarks

Following datasets should not be included in the experiments which are already excluded or commented out from the label-mapping.yaml:

- CoughVID

- TR

- FSD50k

- HFLUNG_Cycle

- ICBHI_Cycle