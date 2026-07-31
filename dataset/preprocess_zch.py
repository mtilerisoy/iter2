import os
import csv
import json
import wave
from pathlib import Path

def create_zchsound_jsonl(csv_path, audio_dir, output_path):
    """
    Processes ZCHSound metadata and audio to create a JSONL file.
    """
    audio_path_obj = Path(audio_dir)
    records_processed = 0

    with open(csv_path, mode='r', encoding='utf-8') as csv_file:
        # Assumes CSV header: filename,label
        reader = csv.DictReader(csv_file)
        
        with open(output_path, mode='w', encoding='utf-8') as jsonl_file:
            for row in reader:
                identifier = row['filename']
                label = row['label']
                
                # Construct the path to the .wav file
                audio_file = audio_path_obj / f"{identifier}.wav"
                
                if not audio_file.exists():
                    print(f"Warning: {audio_file} not found. Skipping entry.")
                    continue

                try:
                    # Extract duration using the standard wave library
                    with wave.open(str(audio_file), 'rb') as wav:
                        frames = wav.getnframes()
                        rate = wav.getframerate()
                        duration = frames / float(rate)
                except Exception as e:
                    print(f"Error reading {audio_file}: {e}")
                    continue

                # Construct the dictionary for this sample
                data_entry = {
                    "identifier": identifier,
                    "label": label,
                    "split": "train",
                    "duration": round(duration, 4),
                    "audio_file": str(audio_file.resolve()) # Full absolute path
                }

                # Write as a single JSON line
                jsonl_file.write(json.dumps(data_entry) + '\n')
                records_processed += 1

    print(f"Successfully processed {records_processed} samples.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update these paths to match your local environment
    INPUT_CSV = "/projects/prjs1635/datasets/zchsound/label.csv"
    AUDIO_DIR = "/projects/prjs1635/datasets/zchsound/clean_audio"
    OUTPUT_JSONL = "dataset/zchsound_train.jsonl"
    # ---------------------

    create_zchsound_jsonl(INPUT_CSV, AUDIO_DIR, OUTPUT_JSONL)