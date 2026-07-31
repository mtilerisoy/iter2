import pandas as pd
import json
import os
import wave

# Configuration - Update these paths
CSV_PATH = '/projects/prjs1635/datasets/circor/training_data.csv'
AUDIO_ROOT = '/projects/prjs1635/datasets/circor/training_data'
OUTPUT_JSONL = 'dataset/circor.jsonl'

def get_wav_duration(file_path):
    """Calculates duration of a wav file in seconds."""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def process_patient_data(csv_path, audio_root, output_file):
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    with open(output_file, 'w') as f:
        for _, row in df.iterrows():
            patient_id = str(row['Patient ID'])
            label = row['Murmur']
            
            # Extract locations (e.g., 'AV+PV+TV+MV' -> ['AV', 'PV', 'TV', 'MV'])
            locations_str = str(row['Recording locations:'])
            if pd.isna(locations_str) or locations_str.lower() == 'nan':
                continue
                
            locations = locations_str.split('+')
            
            for loc in locations:
                # Construct filename based on convention: PatientID_Location.wav
                filename = f"{patient_id}_{loc}.wav"
                full_path = os.path.join(audio_root, filename)
                
                # Check if file exists before processing
                if os.path.exists(full_path):
                    duration = get_wav_duration(full_path)
                    
                    if duration is not None:
                        # Construct the JSON entry
                        entry = {
                            "identifier": patient_id,
                            "label": label,
                            "split": "train",
                            "duration": round(duration, 3),
                            "audio_file": full_path
                        }
                        
                        # Write as a single line in JSONL
                        f.write(json.dumps(entry) + '\n')
                else:
                    print(f"Warning: File not found: {full_path}")

if __name__ == "__main__":
    process_patient_data(CSV_PATH, AUDIO_ROOT, OUTPUT_JSONL)
    print(f"Processing complete! File saved to: {OUTPUT_JSONL}")