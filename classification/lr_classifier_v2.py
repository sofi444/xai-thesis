print('Starting...')

import os
import json
import joblib
import argparse
import pandas as pd
import datetime as dt
import numpy as np
import datetime as dt
import pprint as pp

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

from datasets import load_from_disk

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from tqdm import tqdm



print('Done importing')


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(PROJECT_DIR, "feature_extraction/featureExtraction/output/")
RESPONSES_DIR = os.path.join(PROJECT_DIR, "responses/")
MODELS_DIR = os.path.join(PROJECT_DIR, "classification/models/")
PREDS_DIR = os.path.join(PROJECT_DIR, "classification/preds/")
STATS_DIR = os.path.join(PROJECT_DIR, "classification/stats/")
#SPLITS_DIR = os.path.join(PROJECT_DIR, "classification/split_datasets/coqa_force_aug")
SPLITS_DIR = os.path.join(PROJECT_DIR, "classification/split_datasets/coqa")

#exclude_errors = True
exclude_errors = False


def load_x_y(features_filename, responses_filename):
    
    # load features (x)
    features_df = utils.features.load_features(features_filename)

    # load labels (y)
    labels_df = utils.features.load_labels(responses_filename)

    # merge
    data_df = utils.features.merge_and_filter(features_df, labels_df)

    # balance (false is the minority class)
    if args.balance:
        n_false = len(data_df[data_df['outcome']==False])
        data_df = pd.concat([
            data_df[data_df['outcome']==False],
            data_df[data_df['outcome']==True].sample(n=n_false, random_state=1)
        ])

    # for testing: 15 rows, 15 columns
    #data_df = data_df.iloc[-500:, -500:] # tmp
    #data_df.dropna(axis=1, how='any', inplace=True) # drop columns with NaN values
    #data_df = data_df.loc[:, (data_df != data_df.iloc[0]).any()] # drop constant columns
    print(data_df.shape)
 
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
    
    ''' SCALE FEATURES '''
    data_df = utils.features.scale_features(
        data_df, 
        scaler=MinMaxScaler(
            feature_range=(0, 1)
        )
    )

    nf = len(data_df.columns)-1
    print(f"\nNumber of features before selection: {nf}")
    
    tracker = 0
    ''' REMOVE COLLINEAR FEATURES '''
    if "col" in selection_methods:
        # filter out collinear features
        data_df = utils.features.remove_collinear_features(
            data_df,
            threshold=0.80
        )
        nf = len(data_df.columns)-1
    else:
        tracker = 1

    ''' SELECT K BEST '''
    selected_kbest = utils.features.select_k_best(
        data_df,
        k = 70*nf//100 # number of features to keep
    )
    tmp_df = data_df.copy()[selected_kbest]
    tmp_df['outcome'] = data_df['outcome']
    data_df = tmp_df

    ''' VIF AND PVALUE FILTERING '''
    # remove features with pvalue > 0.05 and vif > 5
    features = data_df.drop(columns=['outcome']).columns
    X = sm.add_constant(data_df.drop(columns=['outcome']))
    y = data_df['outcome']

    _start = dt.datetime.now()
    model = sm.Logit(y, X)
    result = model.fit_regularized(method='l1', alpha=1.2, maxiter=300)
    #result = model.fit(method='lbfgs', maxiter=300, disp=False)
    _end = dt.datetime.now()
    print(f"\nTime for init & fitting: {_end-_start}")
    pvalues = result.pvalues

    print("\nCalculating VIFs...")
    vifs = []
    for i in tqdm(range(X.shape[1])):
        vifs.append(variance_inflation_factor(X.values, i))

    for feature, pvalue, vif in zip(features, pvalues, vifs):
        if pvalue > 0.05 or vif > 5:
            data_df.drop(columns=[feature], inplace=True)

    nf = len(data_df.columns)-1
    print(f"\nNumber of features after pvalue and vif filtering: {nf}")


    if ensemble:
        selected_features = []
        if "rfe" in selection_methods:
            selected_features.append(utils.features.recursive_elimination(
                data_df,
                model=LogisticRegression(random_state=1, max_iter=200),
                n_features = 50*nf//100
            ))
    
        if 'kbest' in selection_methods:
            selected_features.append(utils.features.select_k_best(
                data_df,
                k = 50*nf//100
            ))

        final_set = utils.features.ensemble_selection(
            data_df,
            selected_features,
            type='all'
            )
        
        final_set.append('outcome')
        data_df = data_df[final_set]
        print(f"\nNumber of features after ensemble selection: {len(data_df.columns)-1}")
    
    else: # sequential
        for method in selection_methods.split('-'):
            if method == 'col':
                continue # already done
            if method == 'rfe':
                rfe_features = utils.features.recursive_elimination(
                    data_df,
                    model=LogisticRegression(random_state=1, max_iter=1000),
                    n_features = 30*nf//100 if tracker==1 else 70*nf//100
                )
                tmp_df = data_df.copy()[rfe_features]
                tmp_df['outcome'] = data_df['outcome']
                data_df = tmp_df
                nf = len(data_df.columns)-1
                print(f"\nNumber of features after RFE: {nf}")
                tracker += 1

            elif method == 'kbest':
                kbest_features = utils.features.select_k_best(
                    data_df,
                    k = 30*nf//100 if tracker==1 else 50*nf//100
                )
                tmp_df = data_df.copy()[kbest_features]
                tmp_df['outcome'] = data_df['outcome']
                data_df = tmp_df
                nf = len(data_df.columns)-1
                print(f"\nNumber of features after SelectKBest: {nf}")
                tracker += 1        
    
    return data_df



def prepare_splits(data_df, test_size=0.2, random_state=1):

    if existing_splits: 
        print("\nUsing existing splits")
        # use exact same split as in the SPLITS_DIR
        raw_dataset = load_from_disk(SPLITS_DIR)
        
        if exclude_errors:
            # Load errors map
            with open(os.path.join(PROJECT_DIR, "maps/errors_idx_uuid_map.json"), "r") as f:
                errors_map = json.load(f)
            errors_ids = [int(idx) for idx in errors_map.keys()]
            # Remove from dataset instances with pandas_idx in errors_ids
            raw_dataset['train'] = raw_dataset['train'].filter(
                lambda example: example['pandas_idx'] not in errors_ids
            )

        # eg. train_df contains instances in raw_dataset['train']
        # based on the pandas_idx in raw_dataset['train']['pandas_idx']
        train_df = data_df[data_df.index.isin(raw_dataset['train']['pandas_idx'])]
        test_df = data_df[data_df.index.isin(raw_dataset['test']['pandas_idx'])]
    else:
        print("\nCreating new splits")
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


def save_model(model):
    filepath = os.path.join(MODELS_DIR, f"model_{id}_{run_id}.joblib")
    joblib.dump(model, filepath)


def save_predictions(y_pred, test_idxs):
    preds_filepath = os.path.join(PREDS_DIR, f"preds_{id}_{run_id}.json")
    with open(preds_filepath, "w") as f:
        json.dump(
            #{idx:pred for idx, pred in zip(test_idxs, y_pred.tolist())}, 
            {idx:pred for idx, pred in zip(test_idxs, y_pred)}, 
            f,
            indent=4)
    

def save_stats(stats):
    out_filename = f"{id}_{run_id}_{selection_methods}_{feature_types}_{n_instances}"
    out_filepath = os.path.join(STATS_DIR, f"{out_filename}.json")

    with open(out_filepath, "w") as f:
        json.dump(stats, f, indent=4)


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
    print("\nData ready; starting training")
    X_train = sm.add_constant(X_train)
    model = sm.Logit(y_train, X_train)
    result = model.fit_regularized(method='l1', alpha=1.2, maxiter=300)
    #result = model.fit(method='lbfgs', maxiter=200)

    ''' Evaluation '''
    X_test = sm.add_constant(X_test)
    y_pred_prob = result.predict(X_test)
    y_pred = [True if p > 0.5 else False for p in y_pred_prob]

    print(classification_report(y_test, y_pred))

    ''' Model stats'''
    model_stats = {
        'args': vars(args),
        'coefficients': result.params[1:].to_dict(),
        'intercept': result.params[0],
        'scores': classification_report(y_test, y_pred, output_dict=True)
    }

    ''' Save model, predictions, stats '''
    if args.save:
        save_model(model)
        save_stats(model_stats)
        save_predictions(y_pred, test_idxs)
        
        print(f"\nSaved model, predictions and stats with id: {id}_{run_id}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--features_filename",
                        type=str,
                        help="file with feautures extracted from LLM responses (csv or gz in features/)")
    parser.add_argument("--responses_filename",
                        type=str,
                        help="file with responses; used to extract the gold labels (jsonl in responses/)")
    parser.add_argument("--save",
                        action="store_true",
                        help="include --save to save the model, predictions, stats")
    parser.add_argument("--selection_methods",
                        default="all",
                        choices=["all", "col", "rfe", "kbest", "col-rfe", "col-kbest", "kbest-rfe", "rfe-kbest", "col-rfe-kbest", "col-kbest-rfe"],
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
    parser.add_argument("--balance",
                        action="store_true",
                        help="include --balance to balance the dataset")
    
    args = parser.parse_args()
    print(f"\nArguments:\n{pp.pformat(vars(args))}\n")
    main(args)




