import os
import json
import pandas as pd
import datetime as dt
import numpy as np
import datetime as dt
import joblib
import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append("/Users/q616967/Workspace/thesis/uni/xai-thesis/")
sys.path.append("/mount/studenten-temp1/users/dpgo/xai-thesis/")
sys.path.append("../")
import utils.features

import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_from_disk

print("Done importing")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(PROJECT_DIR, "feature_extraction/featureExtraction/output/")
RESPONSES_DIR = os.path.join(PROJECT_DIR, "responses/")
MODELS_DIR = os.path.join(PROJECT_DIR, "classification/models/")
PREDS_DIR = os.path.join(PROJECT_DIR, "classification/preds/")
STATS_DIR = os.path.join(PROJECT_DIR, "classification/stats/")
SPLITS_DIR = os.path.join(PROJECT_DIR, "classification/split_datasets/coqa")



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
    # create feature set from pre-calculate sources
    if feature_set_sources:
        feature_set = utils.features.get_feature_set(
            source_filenames=feature_set_sources
        )
        if any(" " in feature for feature in feature_set): # then there are interactions
            data_df = utils.features.add_interactions(
                data_df,
                feature_set
            )

        feature_set.append('outcome')
        data_df = data_df[feature_set]
    
    # normalize features
    data_df = utils.features.scale_features(
        data_df, 
        scaler=MinMaxScaler(
            feature_range=(0, 1)
        )
    )

    if "col" in selection_methods:
        # filter out collinear features
        data_df = utils.features.remove_collinear_features(
            data_df,
            threshold=0.8
        )

    if ensemble:
        selected_features = []
        if "rfe" in selection_methods:
            selected_features.append(utils.features.recursive_elimination(
                data_df,
                model=LogisticRegression(random_state=1, max_iter=1000),
                n_features = 10
            ))
    
        if 'kbest' in selection_methods:
            selected_features.append(utils.features.select_k_best(
                data_df,
                k = 10
            ))

        final_set = utils.features.ensemble_selection(
            data_df,
            selected_features,
            type='all'
            )
        
        final_set.append('outcome')
        data_df = data_df[final_set]
    
    else:
        for idx, method in enumerate(selection_methods.split('-')):
            if method == 'col':
                continue # idx == 0; this is done regardless
            if method == 'rfe':
                rfe_features = utils.features.recursive_elimination(
                    data_df,
                    model=LogisticRegression(random_state=1, max_iter=1000),
                    n_features = 300 if idx==1 else 150
                )
                tmp_df = data_df.copy()[rfe_features]
                tmp_df['outcome'] = data_df['outcome']

            elif method == 'kbest':
                kbest_features = utils.features.select_k_best(
                    data_df,
                    k = 300 if idx==1 else 150
                )
                tmp_df = data_df.copy()[kbest_features]
                tmp_df['outcome'] = data_df['outcome']

        data_df = tmp_df
    
    return data_df



def prepare_splits(data_df, test_size=0.2, random_state=1):

    if existing_splits: 
        # use exact same split as in the SPLITS_DIR
        raw_dataset = load_from_disk(SPLITS_DIR)
        # eg. train_df contains instances in raw_dataset['train']
        # based on the pandas_idx in raw_dataset['train']['pandas_idx']
        train_df = data_df[data_df.index.isin(raw_dataset['train']['pandas_idx'])]
        test_df = data_df[data_df.index.isin(raw_dataset['test']['pandas_idx'])]
    else:
        # split into train and test
        train_df, test_df = train_test_split(
            data_df, 
            test_size=test_size, 
            random_state=random_state
        )

    X_train = train_df.drop(columns=['outcome'])
    y_train = train_df['outcome']

    X_test = test_df.drop(columns=['outcome'])
    y_test = test_df['outcome']

    return X_train, y_train, X_test, y_test



def train(X_train, y_train):
    model = LogisticRegression(
        penalty='l2', # l2 defaul
        dual=False, # dual=False when n_samples > n_features. default False (True only for liblinear solver)
        C=1, # inv of regularization strength. smaller values, stronger regularization. default 1.0
        #fit_intercept=True, # add intercept to the decision function. default True
        #intercept_scaling=1, # only used when solver='liblinear' and self.fit_intercept=True. default 1
        class_weight=None, # 'balanced' or dict {class_label: weight}. default None
        random_state=1, # when solver='sag','saga' or 'liblinear'. default None
        solver='lbfgs', # lbfgs default
            # For small datasets, liblinear is a good choice, whereas sag and saga are faster for large ones;
            # For multiclass problems, only newton-cg, sag, saga and lbfgs handle multinomial loss;
            # liblinear is limited to one-versus-rest schemes.
            # newton-cholesky is a good choice for n_samples >> n_features, especially with one-hot encoded categorical features with rare categories. Note that it is limited to binary classification and the one-versus-rest reduction for multiclass classification. Be aware that the memory usage of this solver has a quadratic dependency on n_features because it explicitly computes the Hessian matrix.
            # The choice of the algorithm depends on the penalty chosen. Supported penalties by solver:
            # lbfgs - [l2, None]
            # liblinear - [l1, l2]
            # newton-cg - [l2, None]
            # newton-cholesky - [l2, None]
            # sag - [l2, None]
            # saga - [elasticnet, l1, l2, None]
        max_iter=1000, # default 100
    )
    model.fit(X_train, y_train)
    return model



def save_model(model):
    filepath = os.path.join(MODELS_DIR, f"model_{id}_{run_id}.joblib")
    joblib.dump(model, filepath)



def save_predictions(y_pred, test_idxs):
    preds_filepath = os.path.join(PREDS_DIR, f"preds_{id}_{run_id}.json")
    with open(preds_filepath, "w") as f:
        json.dump(
            {idx:pred for idx, pred in zip(test_idxs, y_pred.tolist())}, 
            f,
            indent=4)

    

def save_stats(stats):
    out_filename = f"{id}_{run_id}_{selection_methods}_{feature_types}_{n_instances}"
    out_filepath = os.path.join(STATS_DIR, f"{out_filename}.json")

    with open(out_filepath, "w") as f:
        json.dump(stats, f, indent=4)



def display_results(y_test, y_pred, show_cm=True):

    print(classification_report(y_test, y_pred))

    if show_cm:
        plt.figure(figsize=(8,8))
        sns.set(font_scale = 1.5)

        cm = confusion_matrix(y_test, y_pred)
        viz = sns.heatmap(
            cm, 
            annot=True, # show numbers in the cells
            fmt='d', # show numbers as integers
            cbar=False, # don't show the color bar
            cmap='flag', # customize color map
            vmax=175 # to get better color contrast
        )
        viz.set_xlabel("Predicted", labelpad=20)
        viz.set_ylabel("Actual", labelpad=20)
        plt.show()



def main(args):

    global id, run_id, existing_splits
    id = args.features_filename.split("/")[-1].split("_")[0]
    run_id = dt.datetime.now().strftime("%H%M")
    existing_splits = args.existing_splits

    global feature_set_sources, selection_methods, feature_types, ensemble
    selection_methods = args.selection_methods
    feature_set_sources = args.feature_set_sources.split(",") if args.feature_set_sources != "" else None
    feature_types = args.feature_types
    ensemble = args.ensemble

    ''' Data '''
    data_df = load_x_y(args.features_filename, args.responses_filename)
    X_train, y_train, X_test, y_test = prepare_splits(data_df, 
                                                      test_size=0.2, 
                                                      random_state=1)

    global n_instances
    n_instances = len(data_df)

    # idxs in test set
    test_idxs = X_test.index.tolist()

    ''' Training '''
    model = train(X_train, y_train)

    ''' Evaluation '''
    y_pred = model.predict(X_test) # numpy array with numpy.bool_ values
    display_results(y_test, y_pred, show_cm=False)

    ''' Model stats'''
    model_stats = {
        'model': model.__class__.__name__,
        'feat_set': selection_methods,
        'feat_types': feature_types,
        'ensemble': ensemble,
        'coefficients': {
            feature: coef for feature, coef 
            in zip(X_train.columns, model.coef_[0])
        },
        'intercept': model.intercept_[0],
        'scores': classification_report(y_test, y_pred, output_dict=True)
    }
    
    ''' Save model, predictions, stats '''
    if args.save:
        save_model(model)
        save_predictions(y_pred, test_idxs)
        save_stats(model_stats)
        print(f"\nSaved model, predictions and stats with id: {id}_{run_id}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--features_filename",
                        type=str,
                        help="file with feautures extracted from LLM responses (csv or gz in output/)")
    parser.add_argument("--responses_filename",
                        type=str,
                        help="file with responses; used to extract the gold labels (jsonl in responses/)")
    parser.add_argument("--save",
                        action="store_true",
                        help="include --save to save the model, predictions, stats")
    parser.add_argument("--selection_methods",
                        default="all",
                        choices=["all", "col", "col-rfe", "col-kbest", "col-rfe-kbest", "col-kbest-rfe"],
                        help="methods to use for feature selection")
    parser.add_argument("--ensemble",
                        action="store_true",
                        help="include --ensemble to use ensemble selection")
    parser.add_argument("--feature_types",
                        type=str,
                        default="all",
                        choices=["all", "trad", "arg"],
                        help="whether the features are from traditional tools or from argadapters, or all")
    parser.add_argument("--feature_set_sources",
                        type=str,
                        default="",
                        help="sources (filenames, comma delimited str) for feature selection (json in stats/)")
    parser.add_argument("--existing_splits",
                        action="store_true",
                        help="include --existing_splits to use existing splits as for transformers models")
    
    args = parser.parse_args()
    print(args)
    main(args)




