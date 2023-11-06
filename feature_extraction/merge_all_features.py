
import os
import pandas as pd
import argparse


# Directories
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(PROJECT_DIR, "feature_extraction/features/")


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--trad_features_file", 
                    type=str, 
                    required=True, 
                    help="File with traditional features (can pass path or filename)") 
parser.add_argument("--arg_features_file",
                    type=str, 
                    required=True,
                    help="File with arg features (can pass path or filename)")
args = parser.parse_args()


# Get run id and make sure it is the same for both files
run_id = args.trad_features_file.split("/")[-1].split("_")[0]
try:
    assert run_id == args.arg_features_file.split("/")[-1].split("_")[0]
except AssertionError:
    print("\n! Run id mismatch; you are trying to merge features from different data")
    exit()

# Set paths (can pass path or filename)
if os.path.exists(args.trad_features_file):
    trad_path = args.trad_features_file
else:
    trad_path = os.path.join(FEATURES_DIR, args.trad_features_file)

if os.path.exists(args.arg_features_file):
    arg_path = args.arg_features_file
else:
    arg_path = os.path.join(FEATURES_DIR, args.arg_features_file)

# Read data
trad_df = pd.read_csv(
    trad_path, compression='gzip'
    ) if trad_path.endswith(".gz") else pd.read_csv(trad_path)

arg_df = pd.read_csv(
    arg_path, compression='gzip'
    ) if arg_path.endswith(".gz") else pd.read_csv(arg_path)

# Merge
merged_df = pd.merge(
    trad_df, arg_df, 
    how='left', 
    left_on='idx', 
    right_on='idx'
)

# Save
merged_df.to_csv(
    os.path.join(FEATURES_DIR, f"{run_id}_all_features.csv.gz"),
    index=False,
    compression='gzip'
)