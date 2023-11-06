
import os
import pandas as pd
import argparse

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--responses_file",
                    type=str,
                    required=True,
                    help="File with responses (can pass path or filename)")
parser.add_argument("--output_dir_name",
                    type=str,
                    required=True,
                    help="Name of output directory (e.g., coqa_force)")

args = parser.parse_args()


# Directories
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses')
SPLITS_DIR = os.path.join(PROJECT_DIR, 'classification/split_datasets')
OUTPUT_DIR = os.path.join(SPLITS_DIR, args.output_dir_name)


# Load data
if os.path.exists(args.responses_file):
    responses_path = args.responses_file
else:
    responses_path = os.path.join(RESPONSES_DIR, args.responses_file)

# Create df
# do not add an index column
data_df = pd.read_json(responses_path, lines=True).drop(columns=['idx', 'uuid', 'parsed'])
data_df['eval'] = data_df['eval'].apply(lambda x: x['outcome'])

# Create splits
train, tmp = train_test_split(data_df, test_size=0.2, random_state=42)
val, test = train_test_split(tmp, test_size=0.5, random_state=42)
train.shape, val.shape, test.shape

# Create datasets
raw_datasets = DatasetDict({'train': Dataset.from_pandas(train),
                            'validation': Dataset.from_pandas(val), 
                            'test': Dataset.from_pandas(test)})

for key in raw_datasets.keys():
    raw_datasets[key] = raw_datasets[key].rename_column("eval", "label")
    raw_datasets[key] = raw_datasets[key].rename_column("__index_level_0__", "pandas_idx")

# Save to disk
# before tokenization, so it doesn't include input_ids, attention_mask, etc.
# need this for knowing which instances are in which split
raw_datasets.save_to_disk(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")