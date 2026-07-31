"""
Preprocessing script for the HF Lung dataset.

Run once to generate the cache:
    python preprocess_hflung.py

Then embed with any model:
    python embed.py --dataset hflung --model dinov2

Audio file naming conventions in the source directory:
    steth_yyyymmdd_HH_MM_ss*.wav   — Littmann 3200 stethoscope
    trunc_yyyy-mm-dd-HH-MM-ss*.wav — HF_Type-1 multichannel recorder

Each audio file has a companion annotation file:
    <stem>_label.txt

Annotation file format (whitespace-delimited, one entry per line):
    <label>  <start_sec>  <end_sec>

Labels:
    I, E          — respiratory phase (inspiration / expiration)
    D             — crackle (discontinuous adventitious sound)
    Wheeze        — wheeze
    Stridor       — stridor
    Rhonchi       — rhonchi

Cycle-extraction strategy
--------------------------
1. Collect all pathology windows (D, Wheeze, Stridor, Rhonchi).
2. For each phase window (I / E):
     - Compute the gap between the phase window and every pathology window.
       Two intervals [a1, a2] and [b1, b2] have
         gap = max(0, max(a1, b1) − min(a2, b2))
       i.e. 0 when they overlap, positive when separated.
     - If the gap to every pathology window is >= OVERLAP_THRESHOLD → Healthy.
3. Pathology cycles: D → Crackle; Wheeze / Stridor / Rhonchi kept as-is.
"""

import pickle
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import json

_PROJECT_ROOT = Path(__file__).resolve().parent

# ── Audio / spectrogram parameters (must match other datasets) ────────────────
SR          = 16000
N_MELS      = 64
FMIN        = 50
FMAX        = 8000
N_FFT       = 1024
HOP_LENGTH  = 512
MIN_SAMPLES = N_FFT   # discard cycles shorter than one FFT window

# A phase window is Healthy only if every pathology window is at least this
# many seconds away (non-overlapping and separated by >= threshold).
OVERLAP_THRESHOLD = 0.5  # seconds

# ── Paths (single source of truth: HFLungDataset) ─────────────────────────────
RAW_AUDIO_DIR   = Path("/projects/prjs1635/datasets/hf_lung/audio/")   # source: original recordings + _label.txt files
SPEC_DIR        = Path("/projects/prjs1635/datasets/hf_lung/spectrograms_cycle/")
CYCLE_AUDIO_DIR = Path("/projects/prjs1635/datasets/hf_lung/audio_cycle/")
METADATA_PATH   = f"{_PROJECT_ROOT}/hflung.jsonl"

# ── Label mappings ────────────────────────────────────────────────────────────
# Labels that indicate a pathological event
PATHOLOGY_LABELS = {"D", "Wheeze", "Stridor", "Rhonchi"}
# Labels that indicate a respiratory phase (used to mine Healthy cycles)
PHASE_LABELS     = {"I", "E"}
# Map raw pathology label → normalised display label
LABEL_MAP = {
    "D":       "Crackle",
    "Wheeze":  "Wheeze",
    "Stridor": "Stridor",
    "Rhonchi": "Rhonchi",
}


# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_mel_spec(audio: np.ndarray) -> np.ndarray:
    """Return a (n_mels × time_frames) float32 spectrogram normalised to [0, 1]."""
    S = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS, fmin=FMIN,
        fmax=FMAX, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    S = librosa.power_to_db(S, ref=np.max)
    if S.max() > S.min():
        return ((S - S.min()) / (S.max() - S.min())).astype(np.float32)
    warnings.warn("Constant spectrogram detected (likely silent audio).")
    return np.zeros_like(S, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Annotation parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_time(s: str) -> float:
    """
    Convert a timestamp string to seconds.

    Accepts both plain floats ("1.5") and HH:MM:SS.mmm notation
    ("00:00:01.500").  Raises ValueError on unrecognised formats.
    """
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected HH:MM:SS.mmm, got {s!r}")
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return float(s)


def parse_annotation(annot_path: Path) -> list[dict]:
    """
    Parse a _label.txt file into a list of dicts with keys label / start / end.

    Expected line format (whitespace-delimited):
        <label>  <start>  <end>

    Timestamps may be plain floats ("1.5") or HH:MM:SS.mmm ("00:00:01.500").
    Lines that cannot be parsed are skipped with a warning.
    """
    entries = []
    with open(annot_path, "r") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                warnings.warn(
                    f"{annot_path.name}:{lineno} — expected 3 fields, got {len(parts)}: {line!r}"
                )
                continue
            label = parts[0]
            try:
                start = _parse_time(parts[1])
                end   = _parse_time(parts[2])
            except ValueError:
                warnings.warn(
                    f"{annot_path.name}:{lineno} — could not parse times: {line!r}"
                )
                continue
            if start >= end:
                warnings.warn(
                    f"{annot_path.name}:{lineno} — start >= end, skipping: {line!r}"
                )
                continue
            entries.append({"label": label, "start": start, "end": end})
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Cycle extraction
# ─────────────────────────────────────────────────────────────────────────────

def _interval_gap(a_start: float, a_end: float,
                   b_start: float, b_end: float) -> float:
    """
    Return the temporal gap between two intervals in seconds.

    Returns 0.0 when the intervals overlap (or touch).
    Returns a positive value when they are separated.
    """
    return max(0.0, max(a_start, b_start) - min(a_end, b_end))


def extract_cycles(annotations: list[dict]) -> list[dict]:
    """
    Apply the cycle-extraction strategy described in the module docstring.

    Returns a list of dicts with keys: label, start, end.
    """
    pathology = [a for a in annotations if a["label"] in PATHOLOGY_LABELS]
    phases    = [a for a in annotations if a["label"] in PHASE_LABELS]

    cycles: list[dict] = []

    # ── Pathology cycles ──────────────────────────────────────────────────────
    for ann in pathology:
        cycles.append({
            "label": LABEL_MAP[ann["label"]],
            "start": ann["start"],
            "end":   ann["end"],
        })

    # ── Healthy cycles from phase windows ────────────────────────────────────
    for ann in phases:
        is_clear = all(
            _interval_gap(ann["start"], ann["end"], p["start"], p["end"])
            >= OVERLAP_THRESHOLD
            for p in pathology
        )
        if is_clear:
            cycles.append({
                "label": "Healthy",
                "start": ann["start"],
                "end":   ann["end"],
            })

    return cycles


# ─────────────────────────────────────────────────────────────────────────────
# Train / test split  (recording-level — no patient IDs available in filenames)
# ─────────────────────────────────────────────────────────────────────────────

def add_recording_split(df: pd.DataFrame,
                        test_ratio: float = 0.20,
                        seed: int = 42) -> pd.DataFrame:
    """
    Assign a train/test split at the *recording* level.
    All cycles from the same recording land in the same split.
    """
    rng        = np.random.default_rng(seed)
    recordings = df["identifier"].unique()
    n_test     = max(1, int(len(recordings) * test_ratio))
    test_recs  = set(rng.choice(recordings, size=n_test, replace=False).tolist())
    df         = df.copy()
    df["split"] = df["identifier"].apply(
        lambda r: "test" if r in test_recs else "train"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    
    CYCLE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(RAW_AUDIO_DIR.glob("*.wav"))
    if not audio_files:
        raise RuntimeError(
            f"No .wav files found in {RAW_AUDIO_DIR}. "
            "Check that RAW_AUDIO_DIR points to the correct location."
        )

    all_cycles: list[dict] = []
    skipped = 0

    for audio_path in tqdm(audio_files, desc="Processing recordings"):
        stem = audio_path.stem

        # Identify device from filename prefix
        if stem.startswith("steth_"):
            device = "steth"
        elif stem.startswith("trunc_"):
            device = "trunc"
        else:
            device = "unknown"

        # Locate companion annotation file
        annot_path = audio_path.parent / f"{stem}_label.txt"
        if not annot_path.exists():
            print(f"\n  [skip] Annotation not found: {annot_path}")
            skipped += 1
            continue

        # Load audio
        try:
            audio, _ = librosa.load(audio_path, sr=SR)
        except Exception as exc:
            print(f"\n  [error] Loading {audio_path}: {exc}")
            skipped += 1
            continue

        # Parse annotations and derive cycles
        annotations = parse_annotation(annot_path)
        if not annotations:
            print(f"\n  [skip] No valid annotations in {annot_path.name}")
            skipped += 1
            continue

        cycles = extract_cycles(annotations)

        for cycle_idx, cycle in enumerate(cycles):
            start_sample = int(cycle["start"] * SR)
            end_sample   = min(int(cycle["end"] * SR), len(audio))
            cycle_audio  = audio[start_sample:end_sample]

            if len(cycle_audio) < MIN_SAMPLES:
                continue

            # ── Save cycle audio ─────────────────────────────────────────────
            audio_file = f"{stem}_{cycle_idx}.wav"
            sf.write(CYCLE_AUDIO_DIR / audio_file, cycle_audio, SR)

            # ── Save mel-spectrogram ─────────────────────────────────────────
            mel       = compute_mel_spec(cycle_audio)
            spec_file = f"{stem}_{cycle_idx}.npy"
            np.save(SPEC_DIR / spec_file, mel)

            all_cycles.append({
                "identifier": stem,
                "device":     device,
                "label":      cycle["label"],
                "start":      cycle["start"],
                "end":        cycle["end"],
                "duration":   cycle["end"] - cycle["start"],
                "audio_file": f"{CYCLE_AUDIO_DIR / audio_file}",
                "spec_file":  f"{SPEC_DIR / spec_file}",
            })

    if not all_cycles:
        raise RuntimeError("No cycles were extracted. Check annotation format and audio files.")

    # save the dict list as a .jsonl file
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        for record in all_cycles:
            # json.dumps does NOT escape forward slashes by default
            # we also use ensure_ascii=False to keep special characters intact
            line = json.dumps(record, ensure_ascii=False)
            f.write(line + '\n')

    print(f"\nDone!")
    print(f"  Cycles processed : {len(all_cycles)}")
    print(f"  Recordings skipped: {skipped}")
    # Fixed CACHE_DIR reference to use the metadata parent path
    print(f"  Cache location   : {METADATA_PATH}")
    
    # meta_df = pd.DataFrame(all_cycles)
    # meta_df = add_recording_split(meta_df)

    # with open(METADATA_PATH, "wb") as fh:
    #     pickle.dump(meta_df, fh)

    print(f"\nDone!")
    print(f"  Recordings processed : {len(audio_files) - skipped} / {len(audio_files)}")
    print(f"  Recordings skipped   : {skipped}")
    print(f"  Cycles extracted     : {len(all_cycles):,}")
    print(f"\n  Label distribution:")
    for lbl, cnt in all_cycles["label"].value_counts().items():
        print(f"    {lbl:12s}: {cnt:,}")
    print(f"\n  Device breakdown:")
    for dev, cnt in all_cycles["device"].value_counts().items():
        print(f"    {dev:12s}: {cnt:,}")
    print(f"\n  Split:")
    for sp, cnt in all_cycles["split"].value_counts().items():
        print(f"    {sp:8s}: {cnt:,} cycles")
    print(f"\n  Cache location: {CYCLE_AUDIO_DIR.resolve()}")


if __name__ == "__main__":
    main()
