import os
import pandas as pd
import json
import numpy as np

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(PROJECT_DIR, "feature_extraction")
RESPONSES_DIR = os.path.join(PROJECT_DIR, "responses/")

features_filepath = os.path.join(FEATURES_DIR, "featureExtraction/output/merged_features.csv")
responses_filepath = os.path.join(RESPONSES_DIR, "formatted_turbo14081857_turbo1508_eval.json")


def load_features(filepath):
    '''
    Load features from .csv file
    Return a pandas dataframe
    '''
    return pd.read_csv(filepath)


def load_labels(filepath):
    '''
    Load evaluated responses (i.e., with predicted labels/outcomes)
    Return map between ids and labels as pandas dataframe
    '''
    with open(filepath, "r") as f:
        responses = json.load(f)

    if filepath.endswith(".json"): # all responses
        idx_label_map = {
            int(idx):res_dict['outcome'] for idx, res_dict in responses.items()
        }
    elif filepath.endswith(".jsonl"): # batch responses
        idx_label_map = {
            int(res_dict['idx']):res_dict['outcome'] for res_dict 
            in responses
        }
    else:
        raise ValueError("Filepath must end with .json or .jsonl")
    
    return pd.DataFrame.from_dict(idx_label_map, orient="index", columns=["outcome"])


def merge_and_filter(features_df, labels_df, only_numeric=True):
    '''
    Merge features and labels dataframes into one dataframe
    Filter out missing values, constant values, etc..
    Returns filtered pandas dataframe
    '''
    try:
        assert len(features_df) == len(labels_df)
    except:
        raise AssertionError(f"Length of features_df ({len(features_df)}) and labels_df ({len(labels_df)}) do not match.")

    data_df = pd.merge(
        features_df, labels_df, left_index=True, right_index=True
    )
    
    data_df.dropna(axis=1, how='any', inplace=True) # drop columns with NaN values
    data_df = data_df.loc[:, (data_df != data_df.iloc[0]).any()] # drop constant columns
    data_df.drop(columns=['idx'], inplace=True) # drop idx column

    if only_numeric:
        if len(data_df.select_dtypes(exclude=['int64', 'float64']).columns) > 1:
            raise ValueError("Found non-numeric columns. The only non-numeric column should be the outcome/label column.")
    
    return data_df
    

def scale_features(data_df):
    '''
    Normalize the values of the features (to the same range/scale)
    Returns normalized pandas dataframe
    '''
    features = [col for col in data_df.columns if col != "outcome"]
    scaler = StandardScaler()
    data_df[features] = scaler.fit_transform(data_df[features])
    
    return data_df


def select_k_best(data_df, k=50):
    '''
    Select k best features according to statistical tests
    Returns names of selected features as array
    '''
    print("Running SelectKBest...")

    selector = SelectKBest(f_classif, k=k)
    selector.fit(
        X = data_df.drop(columns=["outcome"]), 
        y = data_df["outcome"]
    )

    idx_selected_features = selector.get_support(indices=True)
    selected_features = data_df.drop(columns=["outcome"]).columns[idx_selected_features]
    
    print(f"Done! # selected features: {len(selected_features)}")
    return selected_features


def remove_collinear_features(features_df, threshold, target_name='outcome'):
    '''
    Check for collinear features (highly correlated to another feature)
    and remove the one with lower correlation to the target.
    Returns a df which excludes the collinear features.

    features_df: dataframe with features, which includes the target
    threshold: correlation threshold above which to remove features
    target_name: name of the target column
    '''
    print("Removing collinear features...")
    corr_matrix = features_df.corr()
    to_drop = []

    # Iterate through the correlation matrix and compare correlations
    for i in tqdm(range(len(corr_matrix.columns) - 1)):
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            if val >= threshold: # exceeds the threshold
                feature1, feature2 = col.values[0], row.values[0]

                if feature1 in to_drop or feature2 in to_drop:
                    continue
                
                correlation = round(val[0][0], 2)

                # Calculate correlations with outcome for both features
                corr_f1_y = features_df[feature1].corr(features_df[target_name])
                corr_f2_y = features_df[feature2].corr(features_df[target_name])

                # Drop the feature with lower correlation to outcome
                if corr_f1_y <= corr_f2_y:
                    to_drop.append(feature1)
                else:
                    to_drop.append(feature2)
    
    no_coll_df = features_df.drop(columns=to_drop)

    print(f"Done! # selected features: {len(no_coll_df.columns)}")
    return no_coll_df


def sequential_selection(data_df, model, **kwargs):
    '''
    Forward or backward features selection: 
        add (or remove) features sequentially (one by one),
        keep the ones that improve the model the most.
    Returns array of selected features.

    n_features_to_select='auto' or int
    if 'auto', and tol is None (default), the number of features to select is half the number of features.
    if tol is float, then features are selected while the score change does not exceed tol
    '''
    print(f"Running {kwargs['direction']} Sequential Selection...")

    sfs = SequentialFeatureSelector(model,
                                    tol=kwargs['max_score_change'],
                                    direction=kwargs['direction'],
                                    n_features_to_select=kwargs['n_features'],
                                    scoring='accuracy',
                                    cv=kwargs['cv'],
                                    )
    
    sfs.fit(data_df.drop(columns=["outcome"]), data_df["outcome"])
    selected_features = sfs.get_feature_names_out() # array
    
    print(f"Done! # selected features: {len(selected_features)}")
    return selected_features
    

def recursive_elimination(data_df, model, **kwargs):
    '''
    Recursive feature elimination: 
        remove features recursively,
        select the ones that improve the model the most.
    Returns array of selected features.

    n_features_to_select: int or float, default=None
        If None, half of the features are selected. 
        If integer, the parameter is the absolute number of features to select. 
        If float between 0 and 1, it is the fraction of features to select.   
    step: features to remove at each iteration. Default is 1.
    '''
    print(f"Running Recursive Feature Elimination...")
    RFE_selector = RFE(estimator=model,
                       n_features_to_select=kwargs['n_features'],
                       step=1,
                       verbose=0)
    RFE_selector.fit(data_df.drop(columns=["outcome"]), data_df["outcome"])

    selected_features = RFE_selector.get_feature_names_out() # array
    
    print(f"Done! # selected features {len(selected_features)}")
    return selected_features


def ensemble_selection(data_df, selected_features_by_methods, type="all"):
    '''
    Ensemble feature selection: 
        select the features that are selected by most methods.
    Returns list of selected features.

    selected_features_by_methods: list of lists of selected features
    '''
    print(f"Ensembling features...")
    feature_counts = {}
    for features in selected_features_by_methods:
        for feature in features:
            if feature in feature_counts:
                feature_counts[feature] += 1
            else:
                feature_counts[feature] = 1

    if type == "most":
        # 60% of len(selected_features_by_methods), round up
        most = int(np.ceil(len(selected_features_by_methods) * 0.6))
        selected_features = [
            feature for feature, count in feature_counts.items() if count >= most
        ]
    elif type == "all":
        selected_features = [
            feature for feature, count in feature_counts.items() if count == len(selected_features_by_methods)
        ]
    
    print(f"Done! # selected features: {len(selected_features)}")
    return selected_features


    
    # find the features that are selected by len(selected_features_by_methods) methods - 1 methods (i.e., all but one)   
    



if __name__ == "__main__":
    
    map = {
        "collinear": {"function":remove_collinear_features,
                      "kwargs": {"threshold": 0.8}},
        "kbest": {"function":select_k_best,
                  "kwargs": {"k": 100}},
        "seq_forward": {"function":sequential_selection,
                        "kwargs": {"direction": "forward",
                                   "n_features": 100,
                                   "max_score_change": None,
                                   "cv": None}},
        "seq_backward": {"function":sequential_selection,
                         "kwargs": {"direction": "backward",
                                    "n_features": 100,
                                    "max_score_change": None,
                                    "cv": None}},
        "recursive": {"function":recursive_elimination,
                      "kwargs": {"n_features": 100}},
        "ensemble": {"function":ensemble_selection,
                     "kwargs": {"type": "all",
                                "selected_features_by_methods": list()}}
    }
    
    #methods_to_apply = ["collinear", "kbest", "seq_forward", "seq_backward", "recursive", "ensemble"]
    methods_to_apply = ["collinear", "kbest", "recursive"]
    prediction_based_methods = ["seq_forward", "seq_backward", "recursive"]

    # if any of the prediction-based methods are to be applied, we need a model
    if any([method in prediction_based_methods for method in methods_to_apply]): 
        model = LogisticRegression(random_state=1, max_iter=200)
    else:
        model = None

    features_df = load_features(features_filepath)
    labels_df = load_labels(responses_filepath)
    data_df = merge_and_filter(features_df, labels_df)
    data_df = scale_features(data_df)

    print(f"# features before feature selection: {len(data_df.columns)}")

    if "collinear" in methods_to_apply:
        data_df = map["collinear"]["function"](data_df, **map["collinear"]["kwargs"])

    if "kbest" in methods_to_apply:
        selected_features_kbest = map["kbest"]["function"](data_df, **map["kbest"]["kwargs"])
        map["kbest"]["selected_features"] = selected_features_kbest

    if "seq_forward" in methods_to_apply:
        selected_features_forward = map["seq_forward"]["function"](data_df, model, **map["seq_forward"]["kwargs"])
        map["seq_forward"]["selected_features"] = selected_features_forward
    
    if "seq_backward" in methods_to_apply:
        selected_features_backward = map["seq_backward"]["function"](data_df, model, **map["seq_backward"]["kwargs"])
        map["seq_backward"]["selected_features"] = selected_features_backward

    if "recursive" in methods_to_apply:
        selected_features_rfe = map["recursive"]["function"](data_df, model, **map["recursive"]["kwargs"])
        map["recursive"]["selected_features"] = selected_features_rfe
    
    if "ensemble" in methods_to_apply:
        map["ensemble"]["selected_features_by_methods"] = [
            _dict["selected_features"] for method, _dict in map.items() 
            if method in methods_to_apply and method != "ensemble" and method != "collinear"
            ]
        enseble_features = map["ensemble"]["function"](data_df, **map["ensemble"]["kwargs"])
        map["ensemble"]["selected_features"] = enseble_features


    # final feature set + write to file
    final_set = set()
    try:
        final_set.update(map["ensemble"]["selected_features"])
    except:
        for method, _dict in map.items():
            if method in methods_to_apply and method != "collinear":
                final_set.update(_dict["selected_features"])
    
    print(f"# features after feature selection: {len(final_set)}")
    
    out_filepath = os.path.join(FEATURES_DIR, "selected_features.txt")
    with open(out_filepath, "w") as f:
        for feature in final_set:
            f.write(f"{feature}\n")









    
