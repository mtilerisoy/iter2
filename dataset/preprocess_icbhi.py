"""
Preprocessing script for ICBHI respiratory cycle visualization.

Run once to generate the cache:
    python preprocess_cycles.py

Subsequently, launch the Streamlit app:
    streamlit run app.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json

from datasets.icbhi import ICBHIDataset

_DS = ICBHIDataset()

_PROJECT_ROOT = Path(__file__).resolve().parent


# ── Audio / spectrogram parameters (must match dataset.py) ─────────────────
SR          = 16000
N_MELS      = 64
FMIN        = 50
FMAX        = 8000
N_FFT       = 1024
HOP_LENGTH  = 512
MIN_SAMPLES = N_FFT   # discard cycles shorter than one FFT window
# INPUT_DURATION = 5.0  # seconds (matches dataset.py's logic for cropping/padding)

# ── Paths (single source of truth: ICBHIDataset) ────────────────────────────
DATA_ROOT = _DS.data_root
AUDIO_DIR = _DS.audio_dir
SPEC_DIR        = _DS.spec_dir
METADATA_PATH   = f"{_PROJECT_ROOT}/icbhi.jsonl"


def compute_mel_spec(audio: np.ndarray) -> np.ndarray:
    """Compute a normalized mel-spectrogram, matching dataset.py's logic.

    Returns shape (n_mels, time_frames) with values in [0, 1].
    """
    S = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS, fmin=FMIN,
        fmax=FMAX, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    S = librosa.power_to_db(S, ref=np.max)
    if S.max() > S.min():
        return ((S - S.min()) / (S.max() - S.min())).astype(np.float32)
    else:
        warnings.warn("Constant spectrogram detected (likely silent audio).")
        return np.zeros_like(S, dtype=np.float32)

def main() -> None:
    
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Split and Label Information ────────────────────────────────────
    split_path = DATA_ROOT / "ICBHI_challenge_train_test.txt"
    diag_path  = DATA_ROOT / "ICBHI_Challenge_diagnosis.txt"

    # Map full identifier -> 'train'/'test'
    split_df = pd.read_csv(split_path, sep="\t", header=None, names=["id", "split"])
    split_dict = dict(zip(split_df["id"], split_df["split"]))

    # Map patient_id -> diagnosis (e.g., '101' -> 'URTI')
    diag_df = pd.read_csv(diag_path, sep="\t", header=None, names=["patient_id", "label"])
    diag_dict = dict(zip(diag_df["patient_id"].astype(str), diag_df["label"]))

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    if not audio_files:
        raise RuntimeError(
            f"No .wav files found in {AUDIO_DIR}. "
            "Check that AUDIO_DIR points to the correct location."
        )

    all_cycles: list[dict] = []
    skipped = 0

    for audio_path in tqdm(audio_files, total=len(audio_files), desc="Processing recordings"):
        identifier = audio_path.stem
        # Path objects don't have .replace(), so we use with_suffix
        annot_path = audio_path.with_suffix(".txt")

        if not audio_path.exists():
            print(f"\n  [skip] Audio not found: {audio_path}")
            skipped += 1
            continue
        if not annot_path.exists():
            print(f"\n  [skip] Annotation not found: {annot_path}")
            skipped += 1
            continue

        try:
            audio, _ = librosa.load(audio_path, sr=SR)
        except Exception as exc:
            print(f"\n  [error] Loading {audio_path}: {exc}")
            skipped += 1
            continue

        # ── Process LABEL AND SPLIT ─────────────────────────────────────
        # Extract patient ID (the part before the first underscore)
        patient_id = identifier.split('_')[0]
        
        label = diag_dict.get(patient_id, "Unknown")
        split = split_dict.get(identifier, "Unknown")

        # ── Save mel-spectrogram ────────────────────────────────────────
        mel       = compute_mel_spec(audio)  # (n_mels, time_frames)
        spec_file = f"{identifier}_mel.npy"
        np.save(SPEC_DIR / spec_file, mel)

        all_cycles.append({
            "identifier": identifier,
            "label":      label,
            "split":      split,
            "duration":   len(audio) / SR,
            "audio_file": f"{AUDIO_DIR}/{identifier}.wav",
            "spec_file":  f"{SPEC_DIR}/{spec_file}",
        })

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


if __name__ == "__main__":
    main()