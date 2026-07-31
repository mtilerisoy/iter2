import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_JSONL = 'dataset/kauh_train.jsonl'
OUTPUT_JSONL = 'dataset/kauh.jsonl'

def apply_stratified_split(input_file, output_file):
    # 1. Load the JSONL into a DataFrame
    try:
        df = pd.read_json(input_file, lines=True)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Loaded {len(df)} recording samples.")

    # 2. Perform the stratified split
    # test_size=0.33 gives you roughly 1/3 for the test set
    # stratify=df['label'] ensures both sets have the same % of Murmur types
    train_df, test_df = train_test_split(
        df, 
        test_size=0.33, 
        stratify=df['label'], 
        random_state=42  # Seed for reproducibility
    )

    # 3. Update the 'split' field
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['split'] = 'train'
    test_df['split'] = 'test'

    # 4. Combine and save
    final_df = pd.concat([train_df, test_df])
    
    # orient='records' and lines=True converts it back to JSONL format
    final_df.to_json(output_file, orient='records', lines=True)

    # Print summary to verify the stratification worked
    print("\nSplit Summary:")
    print(final_df['split'].value_counts())
    print("\nLabel Distribution per Split:")
    print(pd.crosstab(final_df['split'], final_df['label']))

if __name__ == "__main__":
    apply_stratified_split(INPUT_JSONL, OUTPUT_JSONL)