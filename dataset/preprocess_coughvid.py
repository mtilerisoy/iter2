import os
import wave
import pandas as pd
from sklearn.model_selection import train_test_split

def create_coughvid_dataset(audio_dir, labels_csv_path, output_jsonl_path='coughvid_dataset.jsonl'):
    """
    Creates a unified dataset JSONL from the CoughVID dataset files and ground truth CSV.
    
    Args:
        audio_dir (str): Path to the directory containing .wav files (e.g., 'coughvid/train/').
        labels_csv_path (str): Path to the ground truth CSV.
        output_jsonl_path (str): Output path for the unified JSONL file.
    """
    
    # 1. Load the Ground Truth Labels
    df = pd.read_csv(labels_csv_path)
    
    # Clean the data: Drop rows where our target label is missing (NaN)
    # This is critical for CoughVID as many crowdsourced entries lack complete metadata
    df = df.dropna(subset=['status']).copy()
    
    # 2. Perform a Stratified Split
    # Since 1 UUID = 1 patient session in CoughVID, splitting rows effectively splits patients
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['status'],
        random_state=42 # Set for reproducibility
    )
    
    # Create mapping dictionaries
    split_map = {uuid: 'train' for uuid in train_df['uuid']}
    split_map.update({uuid: 'test' for uuid in test_df['uuid']})
    label_map = dict(zip(df['uuid'], df['status']))

    # 3. Process the Audio Files
    dataset_records = []
    
    # We iterate through the UUIDs that survived the NaN-drop
    for uuid, label in label_map.items():
        split = split_map[uuid]
        
        # Construct the target file path based on the UUID
        file_name = f"{uuid}.wav"
        file_path = os.path.join(audio_dir, file_name)
        
        # Skip if the audio file doesn't actually exist in the folder
        if not os.path.exists(file_path):
            continue
            
        # Extract audio duration using the built-in wave library
        try:
            with wave.open(file_path, 'r') as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            duration = None

        # 4. Append to records
        dataset_records.append({
            'identifier': uuid,
            'label': label,
            'split': split,
            'duration': duration,
            'audio_file': os.path.abspath(file_path) # Absolute path
        })

    # 5. Save to unified JSONL
    final_df = pd.DataFrame(dataset_records)
    
    if final_df.empty:
        print("Error: No records were processed. Please check your folder paths and CSV.")
        return
        
    # Ensure columns match exact order requested
    final_df = final_df[['identifier', 'label', 'split', 'duration', 'audio_file']]
    
    # Export as JSON Lines
    final_df.to_json(output_jsonl_path, orient='records', lines=True)
    print(f"Successfully processed {len(final_df)} audio files. Dataset saved to {output_jsonl_path}")

# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    create_coughvid_dataset(
        audio_dir='/projects/prjs1635/datasets/coughvid/wav/',
        labels_csv_path='/projects/prjs1635/datasets/coughvid/metadata_compiled.csv',
        output_jsonl_path='coughvid.jsonl'
    )