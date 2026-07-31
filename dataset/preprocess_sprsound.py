import os
import json
import librosa

# Configuration paths
base_dir = os.path.join('/projects/prjs1635/datasets/SPRSound', 'Classification')
output_jsonl = 'sprsound.jsonl'

# Map the folder structures to our desired split names
splits_config = [
    {
        'json_dir': 'train_classification_json',
        'wav_dir': 'train_classification_wav',
        'split_name': 'train'
    },
    {
        'json_dir': 'valid_classification_json',
        'wav_dir': 'valid_classification_wav',
        'split_name': 'test'  # Mapping valid to test
    }
]

def build_wav_map(wav_dir):
    """
    Recursively scans the directory and returns a dictionary 
    mapping the filename (without extension) to its absolute path.
    """
    wav_map = {}
    for root, _, files in os.walk(wav_dir):
        for file in files:
            if file.endswith('.wav'):
                file_name_without_ext = os.path.splitext(file)[0]
                wav_map[file_name_without_ext] = os.path.abspath(os.path.join(root, file))
    return wav_map

def generate_sprsound_manifest():
    dataset_samples = []

    print("Scanning SPRSound directories and parsing annotations...")

    for config in splits_config:
        json_root_path = os.path.join(base_dir, config['json_dir'])
        wav_root_path = os.path.join(base_dir, config['wav_dir'])
        split_label = config['split_name']
        
        if not os.path.exists(json_root_path) or not os.path.exists(wav_root_path):
            print(f"Warning: Directories for {config['json_dir']} or {config['wav_dir']} not found. Skipping.")
            continue
            
        # Pre-map all wav files in this split to handle mismatched nested structures
        print(f"Mapping audio files for the '{split_label}' split...")
        wav_map = build_wav_map(wav_root_path)
        
        # Recursively find all JSON files using os.walk
        for root, _, files in os.walk(json_root_path):
            for filename in files:
                if not filename.endswith('.json'):
                    continue
                    
                json_path = os.path.join(root, filename)
                file_name_without_ext = os.path.splitext(filename)[0]
                
                # Check if we mapped a corresponding audio file
                if file_name_without_ext not in wav_map:
                    print(f"Warning: Audio file missing for {file_name_without_ext}. Searched in {wav_root_path}")
                    continue
                    
                wav_path = wav_map[file_name_without_ext]
                
                try:
                    # Read the JSON annotation file
                    with open(json_path, 'r', encoding='utf-8') as f:
                        annotation_data = json.load(f)
                        
                    record_annotation = annotation_data.get('record_annotation')
                    
                    # Discard poor quality samples
                    if record_annotation == "Poor Quality":
                        continue
                        
                    # Extract duration efficiently
                    duration = librosa.get_duration(path=wav_path)
                    
                    # Build the sample dictionary
                    sample_dict = {
                        "identifier": file_name_without_ext,
                        "label": record_annotation,
                        "split": split_label,
                        "duration": duration,
                        "audio_file": wav_path
                    }
                    
                    dataset_samples.append(sample_dict)
                        
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON in file {json_path}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    print(f"Successfully processed {len(dataset_samples)} valid audio files.")
    print(f"Writing output to {output_jsonl}...")

    # Write out to JSONL format
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in dataset_samples:
            f.write(json.dumps(item) + '\n')

    print("Done!")

if __name__ == "__main__":
    generate_sprsound_manifest()