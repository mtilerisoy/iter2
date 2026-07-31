import os
import wave
import pandas as pd
from sklearn.model_selection import train_test_split

def create_bmd_dataset(audio_dir, labels_csv_path, output_jsonl_path='bmd_dataset.jsonl'):
    """
    Creates a unified dataset JSONL from the BMD dataset files and ground truth CSV.
    
    Args:
        audio_dir (str): Path to the directory containing .wav files (e.g., 'bmd/train/').
        labels_csv_path (str): Path to the ground truth CSV.
        output_jsonl_path (str): Output path for the unified JSONL file.
    """
    
    # 1. Load the Ground Truth Labels
    df = pd.read_csv(labels_csv_path)
    
    # 2. Perform a Patient-Level Stratified Split
    # We stratify on the 'N' column (Normal vs Not Normal)
    train_patients, test_patients = train_test_split(
        df,
        test_size=0.2,
        stratify=df['N'],
        random_state=42 # Set for reproducibility
    )
    
    # Create a mapping dictionary for Patient ID -> Split
    split_map = {pid: 'train' for pid in train_patients['patient_id']}
    split_map.update({pid: 'test' for pid in test_patients['patient_id']})
    
    # Create a mapping dictionary for Patient ID -> Label (using the 'N' column)
    label_map = dict(zip(df['patient_id'], df['N']))

    # 3. Process the Audio Files
    dataset_records = []
    
    # Generate the column names to check (recording_1 through recording_8)
    recording_columns = [f'recording_{i}' for i in range(1, 9)]
    
    # Iterate through the patient database
    for _, row in df.iterrows():
        patient_id = row['patient_id']
        label = label_map[patient_id]
        split = split_map[patient_id]
        
        # Iterate horizontally through the recording columns for this specific patient
        for col in recording_columns:
            # Skip if the column doesn't exist or is empty (NaN)
            if col not in row or pd.isna(row[col]):
                continue
                
            identifier = str(row[col]).strip()
            
            # Catch cases where pandas might read an empty string as 'nan'
            if not identifier or identifier.lower() == 'nan':
                continue
                
            # Construct the target file path
            file_name = f"{identifier}.wav"
            file_path = os.path.join(audio_dir, file_name)
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
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
                'label': label,
                'split': split,
                'duration': duration,
                'audio_file': os.path.abspath(file_path) # Absolute path as requested
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
    create_bmd_dataset(
        audio_dir='/projects/prjs1635/datasets/bmd/train/',
        labels_csv_path='/projects/prjs1635/datasets/bmd/train.csv',
        output_jsonl_path='bmd.jsonl'
    )