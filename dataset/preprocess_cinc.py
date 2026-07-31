import os
import json
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

# Base directory for the dataset
base_path = '/projects/prjs1635/datasets/cinc'
train_folders = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']

# Output file path
output_jsonl = 'dataset_manifest.jsonl'

dataset_samples = []
labels_for_split = []

print("Scanning directories and extracting audio metadata...")

for folder in train_folders:
    folder_path = os.path.join(base_path, folder)
    ref_path = os.path.join(folder_path, 'REFERENCE.csv')
    
    if not os.path.exists(ref_path):
        print(f"Warning: {ref_path} not found. Skipping folder.")
        continue
    
    # Load labels
    ref_df = pd.read_csv(ref_path, header=None)
    ref_df.columns = ['filename', 'label']
    
    for _, row in ref_df.iterrows():
        file_name = str(row['filename'])
        label = row['label']
        
        # Resolve absolute path
        wav_path = os.path.abspath(os.path.join(folder_path, file_name + '.wav'))
        
        if os.path.exists(wav_path):
            try:
                # get_duration is significantly faster than loading the full audio array
                duration = librosa.get_duration(path=wav_path)
                
                sample_dict = {
                    "identifier": file_name,
                    "label": label,
                    "duration": duration,
                    "audio_file": wav_path
                }
                
                dataset_samples.append(sample_dict)
                labels_for_split.append(label)
                
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

print(f"Successfully processed {len(dataset_samples)} audio files.")
print("Performing 80/20 stratified split...")

# Perform stratified split (80% train, 20% test)
train_data, test_data = train_test_split(
    dataset_samples, 
    test_size=0.2, 
    stratify=labels_for_split, 
    random_state=42 # Set for reproducibility
)

# Assign split keys
for item in train_data:
    item["split"] = "train"
    
for item in test_data:
    item["split"] = "test"

# Recombine or write them sequentially
all_data_split = train_data + test_data

print(f"Writing output to {output_jsonl}...")

# Write to JSONL format
with open(output_jsonl, 'w', encoding='utf-8') as f:
    for item in all_data_split:
        f.write(json.dumps(item) + '\n')

print("Done!")