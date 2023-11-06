
import os
import json
import gzip
import pandas as pd
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(PROJECT_DIR, 'feature_extraction/features')


run_id = '06111210'


# Load the current features - force TSBS - id: 02111129
current_features_path = os.path.join(FEATURES_DIR, '02111129_all_features.csv.gz')
current_features_df = pd.read_csv(current_features_path, compression='gzip')

# Load the new features - force TSBS - id: 06111000
new_features_path = os.path.join(FEATURES_DIR, '06111000_all_features.csv.gz')
new_features_df = pd.read_csv(new_features_path, compression='gzip')

# Reindex the new features (assign idx from 10000 onwards)
new_features_df['idx'] += 10000

# Merge the features
merged_features_df = pd.concat([current_features_df, new_features_df])
merged_features_df = merged_features_df.fillna(0.0)
merged_features_df = merged_features_df.reset_index(drop=True)

print(f"Current features: {current_features_df.shape}")
print(f"New features: {new_features_df.shape}")
print(f"All features: {merged_features_df.shape}\n")

# Save the merged features
merged_features_path = os.path.join(FEATURES_DIR, f'{run_id}_all_features.csv.gz')
merged_features_df.to_csv(merged_features_path, index=False, compression='gzip')

