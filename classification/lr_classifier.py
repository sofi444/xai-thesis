import os
import json
import pandas as pd
import datetime as dt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import joblib

import argparse

import sys
sys.path.append("/Users/q616967/Workspace/thesis/uni/xai-thesis/")
import utils.features

import matplotlib.pyplot as plt
import seaborn as sns



PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(PROJECT_DIR, "feature_extraction/featureExtraction/output/")
RESPONSES_DIR = os.path.join(PROJECT_DIR, "responses/")
MODELS_DIR = os.path.join(PROJECT_DIR, "classification/models/")
PREDS_DIR = os.path.join(PROJECT_DIR, "classification/preds/")
STATS_DIR = os.path.join(PROJECT_DIR, "classification/stats/")



def load_x_y(features_filename, responses_filename):
    
    # load features (x)
    features_df = utils.features.load_features(features_filename)

    # load labels (y)
    labels_df = utils.features.load_labels(responses_filename)

    # merge
    data_df = utils.features.merge_and_filter(features_df, labels_df)

    # convert labels from 'False'/'True' to 0/1
    #data_df['outcome'] = data_df['outcome'].astype(int)

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

    if "col" in feature_set_type:
        # filter out collinear features
        data_df = utils.features.remove_collinear_features(
            data_df,
            threshold=0.8
        )

    selected_features = []
    if "rfe" in feature_set_type:
        selected_features.append(utils.features.recursive_elimination(
            data_df,
            model=LogisticRegression(random_state=1, max_iter=1000),
            n_features = 300
        ))
    
    if 'kbest' in feature_set_type:
        selected_features.append(utils.features.select_k_best(
            data_df,
            k = 300
        ))

    if "rfe" in feature_set_type or "kbest" in feature_set_type:
        final_set = utils.features.ensemble_selection(
            data_df,
            selected_features,
            type='all'
        )
        final_set.append('outcome')
        data_df = data_df[final_set]

    return data_df




def prepare_splits(data_df, test_size=0.2, random_state=1):
    # split into train and test
    train_df, test_df = train_test_split(data_df, 
                                         test_size=test_size,
                                         random_state=random_state)

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



def save_model(model, id):
    filepath = os.path.join(MODELS_DIR, f"model_{id}.joblib")
    joblib.dump(model, filepath)



def save_predictions(y_pred, test_idxs, id):
    preds_filepath = os.path.join(PREDS_DIR, f"preds_{id}.json")
    with open(preds_filepath, "w") as f:
        json.dump(
            {idx:pred for idx, pred in zip(test_idxs, y_pred.tolist())}, 
            f,
            indent=4)

    

def save_stats(stats):
    out_filename = f"{id}_{feature_set_type}_{n_instances}"
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



def main_sklearn(args):
    global id, feature_set_type
    id = args.features_filename.split("/")[-1].split("_")[0]
    feature_set_type = args.feature_set_type

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
        'feat_set': feature_set_type,
        'coefficients': {
            feature: coef for feature, coef 
            in zip(X_train.columns, model.coef_[0])
        },
        'intercept': model.intercept_[0],
        'scores': classification_report(y_test, y_pred, output_dict=True)
    }
    
    ''' Save model, predictions, stats '''
    if args.save:
        #save_model(model, id)
        #save_predictions(y_pred, test_idxs, id)
        save_stats(model_stats)
        print(f"\nSaved model, predictions and stats with id: {id}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--features_filename",
                        type=str,
                        default="merged_features.csv",
                        help="name of the file with feautures extracted from LLM responses")
    parser.add_argument("--responses_filename",
                        type=str,
                        default="formatted_og03081353_run0808_gpt-default_eval.json",
                        help="name of file with responses; used to extract the gold labels")
    parser.add_argument("--save",
                        action="store_true",
                        help="include --save to save the model and predictions")
    parser.add_argument("--feature_set_type",
                        default="all",
                        choices=["all", "col", "col-rfe", "col-kbest", "col-rfe-kbest"],
                        help="feature set used for training")
    
    args = parser.parse_args()
    main_sklearn(args)




