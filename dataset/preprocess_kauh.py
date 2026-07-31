"""
KAUH lung-sound dataset preprocessing — build mel-spectrogram cache.

Source
------
    /projects/prjs1635/datasets/KAUH/AudioFiles/
        Audio files named:  {F}P{N}_{Diagnosis},{SoundType},{Location},{Age},{Gender}.wav
        F ∈ {B=Bell, D=Diaphragm, E=Extended};  N = patient number 1–112.

Output  (stored under  datasets/kauh/viz_cache/)
------
    spectrograms/   ← one .npy per recording  (n_mels × time_frames, float32 [0,1])
    metadata.pkl    ← pandas DataFrame with one row per recording

Run once; skips if cache exists (override with --force).
"""

import argparse
import pickle
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# ── Paths (single source of truth: KAUHDataset) ─────────────────────────────
KAUH_AUDIO_DIR = Path("/projects/prjs1635/datasets/KAUH/AudioFiles")
METADATA_PATH  = Path("/home/milerisoy/sparsity-scratch/dataset/kauh.jsonl")

# ── Mel-spectrogram parameters (identical to ICBHI / dataset.py) ───────────────
SR         = 16_000
N_MELS     = 64
FMIN       = 50
FMAX       = 8_000
N_FFT      = 1_024
HOP_LENGTH = 512

# ── Constants ──────────────────────────────────────────────────────────────────
FILTER_NAME: dict[str, str] = {
    "B": "Bell (20–200 Hz)",
    "D": "Diaphragm (100–500 Hz)",
    "E": "Extended (50–500 Hz)",
}

# Normalise free-text diagnosis to a consistent label
_DIAG_MAP: dict[str, str] = {
    "n":                             "Normal",
    "normal":                        "Normal",
    "asthma":                        "Asthma",
    "copd":                          "COPD",
    "bron":                          "Bronchitis",
    "heart failure":                 "HF",
    "heart failure + copd":          "HF + COPD",
    "heart failure + lung fibrosis": "HF + Lung Fibrosis",
    "lung fibrosis":                 "LF",
    "plueral effusion":              "PE",
    "pleural effusion":              "PE",
    "pneumonia":                     "Pneumonia",
    "asthma and lung fibrosis":      "Asthma + LF",
    "asthma + lung fibrosis":        "Asthma + LF",
}


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise_diagnosis(raw: str) -> str:
    """Map free-text diagnosis string to a consistent title-case label."""
    key = raw.strip().lower()
    return _DIAG_MAP.get(key, raw.strip().title())


def wc_label(sound_type: str) -> str:
    """
    Derive a wheeze/crackle annotation from the sound-type string.

    Token mapping:
      W             → Wheeze
      C, Crep       → Crackle  (Crep = crepitations = fine crackles)
      W + (C/Crep)  → Both
      anything else → Normal   (includes N, B/Bronchial breathing, etc.)
    """
    tokens = set(sound_type.upper().split())
    has_w = "W" in tokens
    has_c = "C" in tokens or "CREP" in tokens
    if has_w and has_c:  return "Both"
    if has_w:            return "Wheeze"
    if has_c:            return "Crackle"
    return "Normal"


def ie_label(sound_type: str) -> str:
    """
    Derive an inspiratory/expiratory annotation from the sound-type string.

    Token mapping:
      I         → Inspiratory
      E         → Expiratory
      I + E     → Both
      (neither) → Unknown
    """
    tokens = set(sound_type.upper().split())
    has_i = "I" in tokens
    has_e = "E" in tokens
    if has_i and has_e:  return "Both"
    if has_i:            return "Inspiratory"
    if has_e:            return "Expiratory"
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Filename parser
# ─────────────────────────────────────────────────────────────────────────────

_FNAME_RE = re.compile(r'^([BDE])P(\d+)_(.*)')


def parse_filename(fname: str) -> dict | None:
    """
    Parse one KAUH audio filename and return a metadata dict, or None on failure.

    Example input:
        BP1_Asthma,I E W,P L L,70,M.wav
    Parsed as:
        filter_code = "B"
        patient_num = 1
        diagnosis   = "Asthma"
        sound_type  = "I E W"
        location    = "P L L"
        age         = 70
        gender      = "M"

    The comma after the patient prefix separates exactly 5 metadata fields;
    `diagnosis` is everything before the last 4 commas so it can contain "+"
    compound names (e.g., "Heart Failure + COPD").
    """
    stem = Path(fname).stem                   # strip .wav
    m    = _FNAME_RE.match(stem)
    if not m:
        return None

    filter_code = m.group(1)                  # "B" | "D" | "E"
    patient_num = int(m.group(2))             # 1 – 112
    rest        = m.group(3)                  # "Asthma,I E W,P L L,70,M"

    parts = rest.split(",")
    if len(parts) < 5:
        return None

    try:
        age    = int(parts[-2].strip())
    except ValueError:
        return None

    gender     = parts[-1].strip()
    location   = parts[-3].strip()
    sound_type = parts[-4].strip()
    diagnosis  = ",".join(parts[:-4]).strip()   # handles "+" compounds safely

    identifier = f"{filter_code}P{patient_num}"

    return {
        "identifier":     identifier,
        "patient_num":    patient_num,
        # "diagnosis_raw":  diagnosis,
        "diagnosis":      normalise_diagnosis(diagnosis),
        "label":          wc_label(sound_type),
        "ie_label":       ie_label(sound_type),
        "split":          "train",
        "audio_file":     fname,                # filename only; join with KAUH_AUDIO_DIR
        "filter_code":    filter_code,
        "filter_name":    FILTER_NAME.get(filter_code, filter_code),
        "age":            age,
        "gender":         gender,
        "location":       location,
        "sound_type_raw": sound_type,
        
    }



# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KAUH preprocessing: mel-spectrograms + metadata cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cache already exists.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not KAUH_AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio directory not found: {KAUH_AUDIO_DIR}")


    wav_files = sorted(KAUH_AUDIO_DIR.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {KAUH_AUDIO_DIR}")

    print(f"Found {len(wav_files)} WAV files  →  {KAUH_AUDIO_DIR}")

    records = []
    skipped = 0

    for wav_path in tqdm(wav_files, desc="Processing recordings"):
        meta = parse_filename(wav_path.name)
        if meta is None:
            print(f"  ⚠  Cannot parse: {wav_path.name}")
            skipped += 1
            continue

        # ── load audio ────────────────────────────────────────────────────────
        try:
            audio, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        except Exception as exc:
            print(f"  ⚠  Load failed ({wav_path.name}): {exc}")
            skipped += 1
            continue

        meta["duration"] = float(len(audio) / SR)
        

        # # ── mel-spectrogram ───────────────────────────────────────────────────
        # spec     = compute_mel_spec(audio)
        # spec_out = SPEC_DIR / meta["spec_file"]
        # np.save(spec_out, spec)

        records.append(meta)

    if not records:
        raise RuntimeError("No records processed — check audio directory and filenames.")

    # save the dict list as a .jsonl file
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        for record in records:
            # json.dumps does NOT escape forward slashes by default
            # we also use ensure_ascii=False to keep special characters intact
            line = json.dumps(record, ensure_ascii=False)
            f.write(line + '\n')

    print(f"\n✓  Saved metadata → {METADATA_PATH}")
    print(f"   Patients  : {records['patient_num'].nunique()} unique")
    print(f"   Skipped   : {skipped}")
    print(f"\n   WC labels : {records['wc_label'].value_counts().to_dict()}")
    print(f"   IE labels : {records['ie_label'].value_counts().to_dict()}")
    print(f"   Diagnoses : {records['diagnosis'].value_counts().to_dict()}")
    print(f"   Genders   : {records['gender'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
