import os
import glob
import wave
import pandas as pd
from sklearn.model_selection import train_test_split

def create_unified_dataset(audio_dir, labels_csv_path, output_jsonl_path='dataset.jsonl'):
    """
    Creates a unified dataset JSONL from audio files and a ground truth label CSV.
    
    Args:
        audio_dir (str): Path to the directory containing .wav files.
        labels_csv_path (str): Path to the ground truth CSV.
        output_jsonl_path (str): Output path for the unified JSONL file.
    """
    
    # 1. Load the Ground Truth Labels
    # usecols=[0,1] handles the trailing commas present in the original data (e.g., "H002,COPD4,")
    labels_df = pd.read_csv(labels_csv_path, usecols=[0, 1])
    labels_df.columns = ['Patient ID', 'Diagnosis']
    
    # 2. Perform a Patient-Level Stratified Split
    # We split patients to avoid data leakage, stratifying by their diagnosis
    train_patients, test_patients = train_test_split(
        labels_df,
        test_size=0.2,
        stratify=labels_df['Diagnosis'],
        random_state=42 # Set for reproducibility
    )
    
    # Create a mapping dictionary for Patient ID -> Split
    split_map = {pid: 'train' for pid in train_patients['Patient ID']}
    split_map.update({pid: 'test' for pid in test_patients['Patient ID']})
    
    # Create a mapping dictionary for Patient ID -> Label
    label_map = dict(zip(labels_df['Patient ID'], labels_df['Diagnosis']))

    # 3. Process the Audio Files
    dataset_records = []
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    
    if not audio_files:
        print(f"Warning: No .wav files found in {audio_dir}")
        
    for file_path in audio_files:
        filename = os.path.basename(file_path)
        identifier = os.path.splitext(filename)[0] # e.g., H002_L1
        patient_id = identifier.split('_')[0]      # e.g., H002
        
        # Skip files if they don't have a label in the CSV
        if patient_id not in label_map:
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
            'identifier': identifier,
            'label': label_map[patient_id],
            'split': split_map[patient_id],
            'duration': duration,
            'audio_file': os.path.abspath(file_path) # Absolute path as requested
        })

    # 5. Save to unified JSONL
    final_df = pd.DataFrame(dataset_records)
    
    # Ensure columns match exact order requested
    final_df = final_df[['identifier', 'label', 'split', 'duration', 'audio_file']]
    
    # Export as JSON Lines
    final_df.to_json(output_jsonl_path, orient='records', lines=True)
    print(f"Successfully processed {len(final_df)} audio files. Dataset saved to {output_jsonl_path}")

# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    create_unified_dataset(
        audio_dir='/projects/prjs1635/datasets/copd/RespiratoryDatabase@TR/',
        labels_csv_path='/projects/prjs1635/datasets/copd/Labels.csv',
        output_jsonl_path='tr.jsonl'
    )