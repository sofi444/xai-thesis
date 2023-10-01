
import os
import pandas as pd



PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_FEATURES_DIR = os.path.join(PROJECT_DIR, "feature_extraction/featureExtraction/output/")
ARG_FEATURES_DIR = os.path.join(PROJECT_DIR, "feature_extraction/ArgQualityAdapters/")

run_id = "12091031"



old_df = pd.read_csv(
    os.path.join(OLD_FEATURES_DIR, f"{run_id}_features.csv.gz"), 
    compression='gzip'
)

arg_df = pd.read_csv(
    os.path.join(ARG_FEATURES_DIR, f"{run_id}_arg_features_all.csv.gz"),
    compression='gzip'
)

# merge
merged_df = pd.merge(
    old_df, arg_df, 
    how='left', 
    left_on='idx', 
    right_on='idx'
)

# save
merged_df.to_csv(
    os.path.join(PROJECT_DIR, "feature_extraction", f"{run_id}_all_features.csv.gz"),
    index=False,
    compression='gzip'
)