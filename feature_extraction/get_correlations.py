
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.append("/Users/q616967/Workspace/thesis/uni/xai-thesis/")
sys.path.append("/mount/studenten-temp1/users/dpgo/xai-thesis/")
import utils.features



PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(PROJECT_DIR, 'feature_extraction', 'features')
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses')



def load_x_y(features_filename, responses_filename):
    
    # load features (x)
    features_df = utils.features.load_features(features_filename)

    # load labels (y)
    labels_df = utils.features.load_labels(responses_filename)

    # merge
    data_df = utils.features.merge_and_filter(features_df, labels_df)

    # for testing: 15 rows, 15 columns
    #data_df = data_df.iloc[-15:, -15:] # tmp

    # normalize + select features
    data_df = manipulate_features(data_df)

    print(f"Instances: {len(data_df)}\nFeatures: {len(data_df.columns)-1}")

    return data_df


def manipulate_features(data_df):
    # normalize features
    data_df = utils.features.scale_features(
        data_df, 
        scaler=MinMaxScaler(
            feature_range=(0, 1)
        )
    )
    
    return data_df



data_df = load_x_y(
    features_filename=os.path.join(FEATURES_DIR, '12091031_all_features.csv.gz'),
    responses_filename=os.path.join(RESPONSES_DIR, '12091031_parsed_turbo_10000_eval.jsonl')
)


# get correlation matrix
corr_matrix = data_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))


# {feature: [(correlated_feature1, correlation_value1), (correlated_feature2, correlation_value2), ...], ...}
high_correlations = {}
for feature in upper.columns:
    correlated_features = upper.index[upper[feature] > 0.8].tolist()
    if len(correlated_features) > 0:
        high_correlations[feature] = [
            (corr_feature, upper[feature][corr_feature]) for corr_feature in correlated_features
        ]


# save to json
with open(os.path.join(FEATURES_DIR, "all-features_high_correlations.json"), "w") as f:
    json.dump(high_correlations, f, indent=4, sort_keys=True)
