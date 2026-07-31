import os
import json
import pandas as pd
import librosa

# Configuration paths - adjust these to match your environment
dataset_dir = '/projects/prjs1635/datasets/hf_lung'
audio_dir = os.path.join(dataset_dir, 'audio')
csv_file_path = os.path.join(dataset_dir, 'metadata.csv') # Replace with your actual CSV filename
output_jsonl = 'hflung.jsonl'

def generate_jsonl_manifest():
    dataset_samples = []

    print(f"Loading metadata from {csv_file_path}...")
    try:
        # Read the ground truth CSV
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at {csv_file_path}")
        return

    print("Scanning audio files and extracting durations...")
    
    # Iterate through the ground truth labels
    for _, row in df.iterrows():
        file_name = str(row['filename'])
        split = str(row['split'])
        label = str(row['label'])
        
        # Construct the absolute path to the .wav file
        wav_path = os.path.abspath(os.path.join(audio_dir, file_name + '.wav'))
        
        if os.path.exists(wav_path):
            try:
                # Extract duration efficiently
                duration = librosa.get_duration(path=wav_path)
                
                # Build the sample dictionary
                sample_dict = {
                    "identifier": file_name,
                    "label": label,
                    "split": split,
                    "duration": duration,
                    "audio_file": wav_path
                }
                
                dataset_samples.append(sample_dict)
                
            except Exception as e:
                print(f"Error processing audio file {wav_path}: {e}")
        else:
            print(f"Warning: Audio file not found for {file_name}. Expected at {wav_path}")

    print(f"Successfully processed {len(dataset_samples)} audio files.")
    print(f"Writing output to {output_jsonl}...")

    # Write out to JSONL format
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in dataset_samples:
            f.write(json.dumps(item) + '\n')

    print("Done!")

if __name__ == "__main__":
    generate_jsonl_manifest()