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
eval_prune_clap.py        Structured-pruning sweep of the CLAP audio tower (see below).
analyze_prune.py          Turns the sweep CSVs into figures/ (see below).
scripts/knn-*.sh          Slurm batch scripts, one per backbone.
scripts/prune-CLAP.sh     Slurm batch script for the pruning sweep.
scripts/analyze-prune.sh  Slurm batch script for the figures (CPU only).
dataset/*.jsonl           Per-dataset manifests (the data contract, see below).
dataset/preprocess_*.py   One-off scripts that built those manifests from raw archives.
dataset/stratify.py       Helper to add stratified train/test splits to a manifest.
results/knn_*.json        Machine-readable results, one file per backbone.
results/prune_*.csv       One row per (dataset, pruned unit) from the pruning sweep.
figures/<sweep>/          Generated figures + ranked unit table. Never edited by hand.
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

## Structured pruning sweep (`eval_prune_clap.py`)

Which parts of CLAP's audio tower actually carry the pathology signal? The sweep ablates
one structural unit at a time from the HTSAT encoder, re-embeds every recording with the
crippled model, and re-runs the same k-NN probe. The drop in AUROC relative to the intact
model is that unit's contribution.

```bash
sbatch scripts/prune-CLAP.sh --prune_type none     # intact reference row
sbatch scripts/prune-CLAP.sh --prune_type block    # 12 blocks
sbatch scripts/prune-CLAP.sh --prune_type head     # 184 heads
```

`laion/clap-htsat-unfused` is a Swin-style HTSAT: 4 stages of depth `[2, 2, 6, 2]` with
`[4, 8, 16, 32]` heads per stage — **12 blocks, 184 heads**. Ablations are always
single-unit, never cumulative; the model is reloaded conceptually intact for each row.

**Pruning is masking, not surgery.** A head is removed by zeroing its slice of the
self-attention context; a block is removed by replacing it with the identity, so its
residual stream passes through untouched. Weights are never modified, so one in-memory
model serves the whole sweep and every configuration is exactly reversible (verified: the
`none` config and the post-restore forward are bit-identical to the unpruned model).

> Do **not** reach for `attention.prune_heads()` here. Swin-family attention keeps a
> `relative_position_bias_table` sized by head count, and the HF implementation does not
> shrink it, so real head surgery crashes with a shape mismatch. Zeroing the context
> slice was verified bit-identical to zeroing that head's rows of the value projection,
> which is what removing the head means.

Block ablation removes attention **and** the MLP of that block. It is therefore strictly
stronger than zeroing all heads of the same block, which leaves the MLP running.

**Cost.** The sweep re-embeds the same audio 184 times, so decoded and feature-extracted
inputs are cached in RAM once per dataset (~250 KB/recording; the largest dataset is
under 1 GB) and only the GPU forward is repeated. Even so the full head sweep is roughly
1.5 h on one A100 — longer than a single comfortable walltime. Re-submit the same command
with `--resume` and it reads the CSV back, skips every `(dataset, pruning_index)` pair
already recorded, and continues:

```bash
sbatch --time=01:00:00 --mem=20G scripts/prune-CLAP.sh --prune_type head --resume --batch-size 64
```

Rows are flushed as they land, so a job killed by the walltime loses nothing. Peak RSS is
~3.3 GB; 20 GB is ample.

### `results/prune_clap_<prune_type>_<pooling>_seed<seed>.csv`

One row per (dataset, ablated unit). AUROC is the headline metric because these datasets
are heavily imbalanced, and it is threshold-free — accuracy would mostly track the
positive rate.

| column | meaning |
|---|---|
| `dataset` | Config name, e.g. `KAUH`. |
| `seed` | Seed set for `random`/`numpy`/`torch`. Nothing here is stochastic; the column keeps the CSV self-describing. |
| `pruning_type` | `head`, `block`, or `none`. |
| `pruning_index` | Position in the sweep, `0..183` / `0..11`; `-1` for `none`. |
| `pruning_id` | Human-readable unit, `s2.b4.h12` = stage 2, block 4, head 12. |
| `pooling` | `projected` (512-d, shared audio/text space) or `pooled` (768-d, pre-projection). |
| `AUROC` | k-NN AUROC on the test split with that unit removed. |

Compare against the `--prune_type none` row of the same `(dataset, pooling, seed)`.

### Figures (`analyze_prune.py`)

```bash
sbatch scripts/analyze-prune.sh --formats png pdf
```

CPU only, ~30 s, no GPU. Every `results/prune_clap_*.csv` is discovered automatically,
matched to the `none` reference of the same `(pooling, seed)`, and rendered into
`figures/<prune_type>_<pooling>_seed<seed>/`. Add a block sweep later and its folder
appears on the next run with no code change — dense sweeps get the line/heatmap
treatment, sweeps of ≤ 20 units get labelled bars.

| figure | what it answers |
|---|---|
| `01_baseline_auroc` | How good is the intact model on each dataset? |
| `02_effect_heatmap` | The whole sweep at once — dataset × unit. |
| `03_mean_effect` | Mean effect per unit, every dataset's own value behind it. |
| `04_effect_by_block` | Where in the network sensitivity concentrates. |
| `05_per_dataset_profiles` | Do the datasets agree at all? |
| `06_extreme_units` | The extremes, with per-dataset dots as a consistency check. |
| `unit_effects.csv` | Ranked mean/std/worst/best effect per unit. |

Everything is plotted as `AUROC(pruned) − AUROC(intact)`, on a diverging scale with a
neutral midpoint: red = removing it hurts, blue = removing it helps.

**Read these with the noise in mind.** There is one measurement per (dataset, unit) —
no repeats, no confidence intervals — so figure 03 draws a ±1 MAD band and figure 06
plots each dataset as its own dot. A bar whose dots straddle zero is noise, not a
finding. The one effect that survives that scepticism in the head sweep is structural:
in figure 04 the block `s1.b0` is the only one of the twelve whose median effect — and
whole interquartile box — sits below zero. Every other block's median is at or above it.

---

## The regularisation control (`eval_control_clap.py`)

Removing a head does two things at once: it destroys that head's *specific arrangement*
of weights, and it perturbs the network in a generic, capacity-shrinking,
regularisation-flavoured way. A gain from pruning is only interesting if it is the first.
This experiment separates them by leaving the head in place and disturbing only it.

```bash
sbatch scripts/control-CLAP.sh                  # 7 datasets x 6 heads x 21 runs = 889 evals
sbatch scripts/control-CLAP.sh --resume         # continue after a walltime cut
sbatch scripts/analyze-control.sh --formats png pdf
```

| condition | what happens to the head | what it isolates |
|---|---|---|
| `intact` | nothing (one reference row per dataset) | baseline |
| `removed` | attention context zeroed, exactly as in the sweep | the claimed gain |
| `shuffled` ×10 | its own weights permuted **among themselves** | arrangement destroyed; count, magnitudes, location preserved |
| `noise` ×10 | Gaussian noise added, σ = each tensor slice's own std | arrangement preserved; matched-scale perturbation |

A "head" here is all of its parameters: its rows of Q/K/V (weights **and** biases), its
columns of the attention output projection, and its column of the relative-position-bias
table — 18,729 parameters for a stage-1 head. Shuffling is verified to preserve the exact
sorted multiset of every one of those 8 tensors, and every mutation is restored
bit-exactly afterwards, so one in-memory model serves the whole experiment.

Heads are selected **per dataset** from the sweep CSV, because "helpful to remove" is a
per-dataset property: top-3 by positive effect and top-3 by negative effect, fewer when
fewer exist, none when a sign is absent. Trial seeds come from SHA-256 of
`(seed, head, condition, trial)` — never Python's `hash()`, which is salted per process.

### Result

Mean ΔAUROC vs intact, over 7 datasets × 3 heads per side (each control condition
averaged over its 10 draws first):

| heads selected as | `removed` | `shuffled` | `noise` |
|---|---|---|---|
| helpful to remove | **+0.0348** | **+0.0292** | +0.0076 |
| harmful to remove | **−0.0238** | **−0.0177** | +0.0026 |

Head by head, `shuffled` tracks `removed` closely (r = +0.97, slope 0.85) while `noise`
barely tracks it at all (r = +0.48, slope 0.18) — `figures/control_projected_seed42/03_control_vs_removal.png`.
Shuffling lands on the same side of zero as removal for **98%** of the 42 (dataset, head)
pairs; noise manages 64%, barely better than a coin.

Scrambling a head's weights reproduces ~84% of the gain where removal helped and ~74% of
the damage where removal hurt, in both directions. Matched-scale noise, which leaves the
arrangement intact, reproduces ~22% and actually flips sign on the harmful side. The
effect is specific to *how the weights are arranged*, not to perturbing the network.

### Two caveats that bound the claim

1. **The `removed` column is selection-biased.** These heads were chosen because their
   removal moved test AUROC in the sweep, on the same test split — so that column carries
   a winner's curse and overstates the true effect size. The `shuffled` and `noise`
   columns are fresh measurements on already-chosen heads, so the decisive comparison —
   `shuffled` vs `noise` — is *not* affected by the selection. Read the table as a
   contrast between the two controls, not as an unbiased estimate of the pruning gain.
2. **Noise at σ is a milder perturbation than removal.** On a probe batch, relative
   embedding shift was 0.076 for noise against 0.119 for removal and 0.146 for shuffling.
   So some of noise's smaller effect may be that it disturbs the representation less, not
   only that it preserves arrangement. Closing that gap means scaling noise until its
   representation shift matches removal's: `--noise-scale` does exactly this, and writes
   its own CSV since the scale is part of the condition.

---

<!-- ## What is special about these heads? (`eval_head_properties.py`) -->

<!-- The sweep locates the heads and the control shows the effect is real and
arrangement-specific. This asks what the heads *are*: measure standard properties of all
184 and test which ones separate helpful-to-remove from harmful-to-remove.

```bash
sbatch scripts/head-properties-CLAP.sh          # ~3 min, one forward pass per corpus
sbatch scripts/analyze-headprops.sh --formats png pdf
```

| property family | measures |
|---|---|
| static (weights only) | `w_norm`, `w_norm_qkv`, `w_norm_out`, `w_rms`, `w_kurtosis`, `w_participation`, `w_max_over_rms`, `ov_spectral` (spectral norm of the head's OV circuit `W_out[:,h] @ W_v[h,:]`) |
| activation, per corpus | `ctx_rms` (context length per token), `contrib_rms` (length of what the head writes into the residual stream), `energy_share` (its share of the block's written energy) |
| domain contrast | `generality` = general − medical energy share; `contrib_ratio` = log2(general / medical contribution) |

Corpora: the seven medical test splits, plus **FSD50k as a general-audio probe** — CLAP's
pretraining domain, and the point of the exercise.

### Result: no property separates the two groups

Per dataset (184 heads × 7 datasets), after z-scoring each property **within its stage**:
every property has |mean ρ| ≤ 0.065 and AUC in 0.48–0.545, and **not one of the fourteen
has a consistent sign across all seven datasets**. This is the "expected if noise" branch.

Against the cross-dataset **mean** effect (less noisy target, so more power), one property
does stand out, but not in the hypothesised way:

| property | ρ (within stage) | p | p after Bonferroni ×14 |
|---|---|---|---|
| general-audio energy share | **+0.20** | 0.007 | 0.09 |
| medical energy share | +0.17 | 0.020 | 0.28 |
| log2(general / medical contribution) | −0.07 | 0.38 | 1.0 |
| generality (general − medical share) | +0.02 | 0.76 | 1.0 |
| every static weight property | \|ρ\| ≤ 0.06 | ≥ 0.43 | 1.0 |

Read together, these say **loudness, not domain**. Heads that write more energy — on
*either* corpus — are somewhat more likely to be worth removing, but the general-vs-medical
*contrast* predicts nothing at all. Nothing survives correction for testing 14 properties.

**And the smoking gun is not there because the ammunition is not there.** Across the 184
heads, general-audio and medical energy share correlate at **r = 0.95**. CLAP's heads are
essentially not domain-specialised in energy terms: there is no population of
general-audio-only heads for the contrast to find. In the top-20 group profile the heads
whose removal helps are, if anything, *relatively louder on medical audio than on general*
(`03_group_profiles.png`) — the opposite of the "general-audio features mislead the
medical probe" story. -->
## What is special about these heads? (`eval_head_properties.py`)

### The selected heads *are* distinct from ordinary heads

Comparing helps-to-remove and hurts-to-remove against **each other** finds nothing.
Comparing each against **the other 178 heads** does. Using the same selection rule as the
control (top-3 of each sign, per dataset) and the unit that rule implies —
(dataset, head), because two heads are helpful for one dataset and harmful for another —
gives 21 helps rows, 21 hurts rows and 1246 rest rows. Each group's median is compared
against the central 95% of 10,000 random selections of the same size, drawn per dataset,
which needs no independence assumption:

| group | property | median z | Cliff's δ | p (permutation) | p after Bonferroni ×42 |
|---|---|---|---|---|---|
| **either** | medical energy share | **+0.35** | +0.29 | 0.0002 | **0.008** ✓ |
| **either** | medical contribution RMS | **+0.43** | +0.30 | 0.0010 | **0.042** ✓ |
| **either** | general-audio energy share | **+0.20** | +0.24 | 0.0009 | **0.038** ✓ |
| helps | general-audio energy share | +0.33 | +0.23 | 0.0040 | 0.17 |
| hurts | medical contribution RMS | +0.47 | +0.43 | 0.0056 | 0.24 |
| either | weight norm | −0.25 | −0.19 | 0.032 | 1.0 |

**Heads that matter are louder than heads that don't — in either direction.** Pooling both
signs (`either`, n=42) is what reaches significance, and the helps and hurts groups shift
the *same* way on every activation measure. So activation magnitude predicts **whether a
head matters at all**, not **which way it matters**. Cliff's δ ≈ +0.29 means a randomly
chosen selected head is louder than a randomly chosen ordinary head about 65% of the time
— a real but moderate separation, not a clean split.

Neither group separately survives correction (n=21 each), and no static weight property
survives in any grouping. This is consistent with everything upstream: loudness buys
influence, the *arrangement* decides the sign, and no descriptor here captures arrangement.

### How to read each figure

`figures/head_properties_projected_seed42/`

**`01_property_ranking`** — y: the 14 properties. x: Spearman ρ between the property
(across 184 heads) and those heads' ΔAUROC, **averaged over the 7 datasets**; the grey
line is the min–max range across those datasets. Left panel z-scores each property within
its stage; right panel does not. Blue = higher property means removal helps more, orange =
hurts more. *Read the line, not the bar*: every property's range straddles zero, so no
property is consistent across datasets. Bars ≤ 0.08 are negligible — ρ = 0.07 explains
about 0.5% of the rank variance.

**`02_generality_vs_effect`** — one point per head (184); shape = stage; y = mean ΔAUROC
over the 7 datasets from removing that head; x = the property, z-scored within stage. Grey
line is a least-squares fit, and each panel's title carries the Spearman ρ and its
uncorrected p. Left is *how loud* the head is, right is *what it is loud on*. The left
tilts up (ρ = +0.20, p = 0.007 raw, 0.09 after correcting for 14 properties); the right is
flat (ρ = −0.07, p = 0.4). Loudness relates weakly; domain preference not at all.

**`03_group_profiles`** — y: properties. Two bars each: the top-20 heads by mean effect
(blue) and the bottom-20 (orange). x: mean z-score within stage, so 0 is the average head
of the same stage; whiskers are ±1 standard error. Compare each bar to 0 *and* to the
other bar. Both bars point the same way on the activation properties — that is the
figure's message: these properties do not discriminate direction. It carries no
significance test; `05` supplies that.

**`04_stage_confound`** — five panels (four properties and the pruning effect itself),
each a box plot across the four HTSAT stages. Box = interquartile range, line = median,
whiskers = 1.5×IQR, outliers hidden. Weight norm climbs 6.8 → 11.7 and OV spectral norm
0.44 → 2.14 from stage 0 to 3. This is *why* every other statistic is computed within
stage: without that, "property predicts effect" could be nothing but "stage predicts both".

**`05_selected_vs_rest`** — y: properties. One panel per group: top-3-helps per dataset
(n = 21 dataset-head rows), top-3-hurts (n = 21), and both pooled (n = 42). x: the group's
**median z-score within (dataset, stage)** — 0 is an ordinary head from the same dataset
and stage. The grey band is the central 95% of 10,000 random same-size selections: **a dot
inside the band is what chance already produces**, a dot outside it is a real shift. A star
marks survival of Bonferroni correction over all 42 tests. The three starred rows in the
pooled panel are the finding above.

**`head_features.csv`** has the per-head numbers, `selected_vs_rest.csv` the group tests,
`property_stats_per_dataset.csv` and `property_stats_mean_effect.csv` the correlations.

### Why this is not proof that head properties are irrelevant

The target variable is noisy. Per-dataset single-head effects sit near the ±0.004 MAD
noise floor established by the sweep, which attenuates every correlation toward zero. The
control experiment showed the *extreme* heads' effects are real (shuffling reproduces
removal at r = +0.97), so the honest conclusion is that **these standard descriptors fail
to predict the effect at the signal-to-noise available** — not that no property could.
A stronger test would need repeated-split effect estimates to raise the target's SNR.

### Assumptions and design choices

Everything here that was a judgement call rather than a given:

1. **Split into two scripts**, `eval_head_properties.py` (measure → `results/head_*.csv`)
   and `analyze_head_properties.py` (test + figures), matching the repo's eval/analyze
   split rather than the single script suggested.
2. **Activations need forward passes**, not the stored embeddings — an embedding carries
   no per-head information. It stays cheap because all 184 heads are instrumented at once:
   **one pass per corpus, 8 total, ~3 min**, versus 184 passes if done per head.
3. **FSD50k is used as a probe corpus, not a task.** The path comes from
   `label-mapping-FSD50k.yaml`, but its `label_map` (2 of ~200 classes) defines a binary
   task; for a general-audio probe the breadth matters, so **all classes are kept**
   (`--general-labels mapped` restores the 2-class version). Subsampled to 1000
   recordings, deterministic under the seed. This is the one place FSD50k enters the
   project despite the "excluded datasets" note below — as a probe stimulus, never as an
   evaluation task.
4. **Medical corpora are probed on the `test` split**, where the sweep effect was measured,
   and the seven are averaged into one number per head.
5. **A head's "activation" is defined twice**: the context it produces, and what it writes
   into the residual stream (context × that head's columns of the output projection). The
   latter is the meaningful one and is what `energy_share` uses.
6. **`energy_share` normalises within the block**, which removes each corpus's overall
   loudness and each stage's width and head count. It sums squared per-head norms, so it
   ignores cross-head cancellation — a share, not an exact decomposition.
7. **Within-stage z-scoring is the load-bearing control.** Head size, weight norm,
   activation scale and pruning effect all vary by stage (`04_stage_confound.png`), so a
   raw correlation can be nothing but "stage predicts both". Raw numbers are reported
   alongside so the size of the confound stays visible.
8. **Two target variables**: per-dataset effect (7 tests, honest about disagreement) and
   the cross-dataset mean (more power, less detail). Both are reported.
9. **Bonferroni correction over the 14 properties**, because this is a screen. The
   uncorrected p = 0.007 is reported too, and neither is treated as a finding.
10. **AUC groups** are the top/bottom 20 heads by effect (`--group-size`), an arbitrary
    but pre-stated cut.



## Does head-harmfulness transfer across tasks? (`analyze_transfer.py`, `eval_family_transfer_clap.py`)

Every earlier experiment treated "the same head helps task A and hurts task B" as the
nuisance that stopped anything correlating. This makes it the measurement. Two tests, one
free and one cheap:

```bash
sbatch scripts/analyze-transfer.sh --formats png pdf      # re-analysis, no forward passes
sbatch scripts/family-transfer-CLAP.sh                    # held-out check, ~9 min
sbatch scripts/analyze-famtransfer.sh --formats png pdf
```

Families are anatomical and were fixed before looking at any result: **cardiac** = BMD
(Mit/Pul/Aor/Tri sites), CinC, CirCor (MV/AV/PV/TV), ZCHSound; **respiratory** = ICBHI,
KAUH, SPRSound.

### 1. The cross-task correlation matrix — no shared structure

Correlating each dataset's 184-head effect vector against every other's:

- largest |ρ| anywhere is **0.151**; **0 of 21 pairs** survive Bonferroni
- within-family mean ρ **+0.016** vs across-family **−0.023**, gap +0.039,
  **exact permutation p = 0.31** over all 35 relabellings
- the dendrogram does not recover the families — ZCHSound (cardiac) clusters with the
  respiratory datasets

This is the uniform-near-zero branch. Ranking heads by "removal helps" on one dataset
tells you essentially nothing about another.

Restricting to each dataset's signal-carrying tail (`05_transfer_matrix.png`: take the
top-10 heads on A, measure them on B) does not rescue it, and reveals the reason the
first test looked so flat — **the matrix has a strong column structure, not a block
structure**. Transfer into ICBHI averages **+0.016** and into KAUH **+0.009** regardless
of where the heads came from, while transfer into CirCor, BMD and CinC is negative
regardless. Some datasets simply move upward under almost any head removal.

Controlling for that per-target main effect, 6 of 7 targets do favour same-family
selection (mean gap +0.004, Wilcoxon p = 0.047, sign test p = 0.125) — but that test was
run *after* the pooled one failed, and one dataset dominates it. It is a hint, not a
result.

### 2. The held-out test — selected head sets do not beat random ones

For each target dataset, three sets of 10 heads are chosen **without ever using that
dataset**, removed all at once, and scored on it; a random 10-head set is the floor.

| target | same family | other family | all others | random (mean of 10) |
|---|---|---|---|---|
| BMD | −0.025 | +0.027 | −0.000 | +0.002 |
| CinC | +0.013 | +0.012 | −0.003 | +0.003 |
| CirCor | +0.003 | −0.012 | −0.007 | −0.004 |
| ICBHI | +0.087 | +0.096 | −0.022 | +0.022 |
| KAUH | −0.043 | −0.043 | +0.012 | +0.004 |
| SPRSound | −0.012 | −0.013 | −0.023 | −0.002 |
| ZCHSound | +0.024 | +0.050 | +0.022 | +0.012 |
| **mean** | **+0.007** | **+0.017** | **−0.003** | **+0.005** |

Within-target paired tests: same-vs-other family **4/7, p = 0.69**; same-family vs random
**4/7, p = 0.94**; universal vs random **2/7, p = 0.47**. In `01_transfer_per_target.png`
essentially every selected set lands **inside** the random band.

**Head sets selected on other datasets transfer no better than random head sets** — not
within family, not universally.

### 3. The control this project had been missing

The random condition is the first time a *k*-head removal was compared against removing
*k* arbitrary heads on the same dataset, and it is sobering:

| dataset | ΔAUROC from 10 random heads |
|---|---|
| ICBHI | **+0.022 ± 0.049** (range −0.052 … +0.097) |
| ZCHSound | +0.012 ± 0.015 |
| KAUH | +0.004 ± 0.027 |
| CirCor | −0.004 ± 0.014 |

ICBHI's k-NN AUROC moves by up to ±0.10 when ten *arbitrary* heads are deleted. So a large
part of what the sweep recorded as "removing this head helps ICBHI" is **ICBHI's
susceptibility to almost any perturbation**, not a property of the head.

**How this squares with the shuffle control.** That experiment showed that destroying a
*specific* head's arrangement reproduces its removal effect (r = +0.97). That remains
true — but it was measured on the same fixed test split, and a susceptible dataset would
reproduce it too: any similar perturbation of the same head moves the same split the same
way. The control established the effect is a reproducible function of
**(head, dataset, split)**; it never established it generalises beyond that split, and
this experiment shows it does not generalise across datasets.

**What would settle it**: re-run the sweep under a different train/test split (or bootstrap
the test set) and correlate the two effect vectors *within* each dataset. That separates
split-specific movement from a real head-by-dataset interaction, and costs about one more
sweep (~45 min). Until that is done, the per-head effects should be described as
split-specific, and the strong claims in the sections above should be read with the random
floor in mind.






## Split-stability: does the per-head effect replicate? (`--split-direction reverse`)

The transfer experiments left one question load-bearing: is "this head helps this dataset"
a property of the (head, dataset) pair, or of the particular recordings that happened to be
scored? This settles it using only the splits the repo already has, by swapping their roles:

```bash
sbatch scripts/prune-CLAP.sh --prune_type none --split-direction reverse --batch-size 64
sbatch --time=01:00:00 scripts/prune-CLAP.sh --prune_type head --split-direction reverse --batch-size 64
sbatch scripts/analyze-splitstab.sh --formats png pdf
```

| direction | k-NN fitted on | scored on |
|---|---|---|
| forward | train | test |
| reverse | test | train |

No new split is invented and no new embedding is computed — the same cached features are
reused and only the classifier's roles change. The two AUROCs are therefore measured on
**disjoint recordings**, which is what makes the comparison a replication test.

### Result: the effect vectors barely agree

| dataset | ρ (all 184 heads) | p | ρ (forward extremes) |
|---|---|---|---|
| SPRSound | +0.260 | 0.0004 | +0.396 |
| KAUH | +0.224 | 0.002 | +0.275 |
| CirCor | +0.195 | 0.008 | +0.388 |
| ICBHI | +0.088 | 0.23 | +0.036 |
| BMD | +0.061 | 0.41 | +0.120 |
| ZCHSound | +0.059 | 0.42 | +0.098 |
| CinC | −0.055 | 0.46 | −0.117 |

Mean ρ **+0.119**, median +0.088. Only **3 of 7** clear the |ρ| > 0.146 needed to be
distinguishable from zero at n = 184, and none comes near the ρ > 0.5 that would mark a
stable head×task interaction. Correlations are attenuated by noise in either direction, so
these are lower bounds — but they are nowhere near the bar.

### The decisive read: what survives, in AUROC units

Take the 10 heads the forward split calls most helpful, read them on the reverse split,
and compare against the mean of random 10-head sets drawn from the same reverse column
(10,000 draws). Selection uses forward only, so the reverse column is out of sample.

| dataset | forward "helps" | reverse | above random floor | survives | p |
|---|---|---|---|---|---|
| ICBHI | **+0.0719** | +0.0034 | +0.0078 | 11% | 0.060 |
| KAUH | +0.0384 | +0.0014 | +0.0021 | 6% | 0.28 |
| BMD | +0.0257 | −0.0068 | −0.0012 | — | 0.72 |
| ZCHSound | +0.0189 | −0.0017 | −0.0004 | — | 0.64 |
| CirCor | +0.0108 | −0.0004 | +0.0029 | 27% | 0.046 |
| CinC | +0.0085 | +0.0015 | +0.0003 | 4% | 0.30 |
| SPRSound | +0.0085 | +0.0011 | −0.0000 | 0% | 0.53 |

**"Removal helps" does not replicate.** ICBHI's headline +0.072 becomes +0.003 — about a
tenth of it, and not significant. One dataset of seven reaches p < 0.05 uncorrected, none
after correcting for the 14 tests. The gains were a winner's curse: heads selected because
they moved *those* recordings.

The other side behaves differently:

| dataset | forward "hurts" | reverse | above floor | survives | p |
|---|---|---|---|---|---|
| SPRSound | −0.0083 | −0.0049 | −0.0060 | 72% | **0.0004** |
| CirCor | −0.0236 | −0.0082 | −0.0049 | 21% | **0.003** |
| KAUH | −0.0178 | −0.0080 | −0.0072 | 41% | 0.040 |
| BMD | −0.0394 | −0.0109 | −0.0054 | 14% | 0.012 |
| ZCHSound | −0.0054 | −0.0012 | +0.0001 | — | 0.55 |
| CinC | −0.0134 | +0.0023 | +0.0010 | — | 0.94 |
| ICBHI | −0.0157 | +0.0029 | +0.0072 | — | 0.92 |

**"Removal hurts" partially replicates**: 4 of 7 datasets at p < 0.05, and SPRSound and
CirCor survive Bonferroni over all 14 tests. Heads that carry pathology signal keep
costing AUROC when deleted, on recordings never used to pick them.

### What this means for the claims

The honest, narrower statement the evidence supports:

- **There are heads whose removal reliably damages a given task.** That replicates
  out-of-sample on 4 of 7 datasets and is a genuine head×task interaction.
- **There are no heads whose removal reliably improves the probe.** Every such gain
  measured here is split-specific. The earlier "removing this head helps ICBHI by 0.07"
  is not a property of the head.
- This retro-explains the two preceding experiments. Family transfer failed because the
  sets were built from "helps" heads, i.e. from noise. And the shuffle control's r = +0.97
  was measured on the fixed forward split — it showed the effect is a reproducible function
  of (head, dataset, split), exactly as flagged, and the split term turns out to dominate
  on the helps side.
- The **arrangement** result still stands as stated: scrambling a head reproduces its
  removal effect, whatever that effect is. It was never evidence that the effect
  generalises.

Anything written up should lead with the harmful direction, quote out-of-sample numbers
from the reverse split rather than the forward selection, and drop the pruning-gain
framing entirely.

**Figures** (`figures/split_stability_projected_seed42/`): `01_split_scatter` is one panel
per dataset, each head's forward effect against its reverse effect with the extremes
highlighted — the diagonal is perfect agreement, and the clouds are round.
`02_split_rho` ranks the ρ values against the n = 184 significance band. `03_extreme_readout`
pairs each selection with its out-of-sample re-read. `04_effect_survival` is the table
above in AUROC units, with the random-head floor marked and stars for p < 0.05.

---

<!-- ---

## Final Remarks

Following datasets should not be included in the experiments which are already excluded or commented out from the label-mapping.yaml:

- CoughVID

- TR

- FSD50k

- HFLUNG_Cycle

- ICBHI_Cycle -->