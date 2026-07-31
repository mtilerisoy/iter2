import pandas as pd
import soundfile as sf
import json
import os

def get_audio_duration(file_path):
    """Reads the duration of a .wav file in seconds."""
    try:
        # sf.info reads just the metadata, which is much faster than loading the whole audio array
        return sf.info(file_path).duration
    except Exception as e:
        print(f"Warning: Could not read duration for {file_path}. Error: {e}")
        return None

def process_dataset(input_csv, output_jsonl):
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # 1. Remove the "mids" field
    if 'mids' in df.columns:
        df = df.drop(columns=['mids'])
        
    # 2. Rename "fname" to "identifier"
    df = df.rename(columns={'fname': 'identifier'})
    
    # 3. Create the "audio_file" field
    prefix = "/projects/prjs1635/datasets/fsd50k/FSD50K.eval_audio/"
    # Cast identifier to string to ensure clean concatenation
    df['audio_file'] = prefix + df['identifier'].astype(str) + ".wav"
    
    # 4. Keep first occurrence of "labels" and rename to "label"
    # Splits the string by comma and grabs the first element
    df['label'] = df['labels'].apply(lambda x: x.split(',')[0] if pd.notnull(x) else x)
    df = df.drop(columns=['labels'])
    
    # 5. Add "split" field with value "test"
    df['split'] = "test"
    
    # 6. Record the duration of the .wav file in seconds
    print("Extracting audio durations... (this may take a moment depending on dataset size)")
    df['duration'] = df['audio_file'].apply(get_audio_duration)
    
    # 7. Reorder fields to match the requested output
    field_order = ['identifier', 'label', 'split', 'duration', 'audio_file']
    df = df[field_order]
    
    # 8. Save modified dataframe as JSONL
    df.to_json(output_jsonl, orient='records', lines=True)
    print(f"Processing complete! Saved to {output_jsonl}")

# --- Execution ---
if __name__ == "__main__":
    # Replace these paths with your actual file names
    INPUT_FILE = "/projects/prjs1635/datasets/fsd50k/FSD50K.ground_truth/eval.csv"
    OUTPUT_FILE = "fsd50k.jsonl"
    
    process_dataset(INPUT_FILE, OUTPUT_FILE)