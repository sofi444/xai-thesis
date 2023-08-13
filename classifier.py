import os
import json
import pandas as pd
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import statsmodels.api as sm

import joblib

import argparse



PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(PROJECT_DIR, "feature_extraction/featureExtraction/output/")
RESPONSES_DIR = os.path.join(PROJECT_DIR, "responses/")
MODELS_DIR = os.path.join(PROJECT_DIR, "models/")
PREDS_DIR = os.path.join(PROJECT_DIR, "classifier_preds/")



def load_x_y(features_filename, responses_filename):
    features_df = pd.read_csv(os.path.join(FEATURES_DIR, features_filename), 
                            index_col=0
                            ).dropna(axis=1).drop(columns=['response', 'filename', 'filename_taaled'])

    # drop all columns with only zeros
    features_df = features_df.loc[:, (features_df != 0).any(axis=0)] # tmp

    with open(os.path.join(RESPONSES_DIR, responses_filename), "r") as f:
        responses = json.load(f)

    idx_outcome_dict = {int(idx):res_dict['outcome'] for idx, res_dict in responses.items()}

    # sample version; only first 5 resposnes
    idx_outcome_dict = {idx:outcome for idx, outcome in idx_outcome_dict.items() if idx < 5} # tmp

    labels_df = pd.DataFrame.from_dict(idx_outcome_dict, columns=['outcome'], orient='index')

    try:
        assert len(features_df) == len(labels_df)
    except AssertionError:
        print("Length mismatch between features and labels")

    # merge on index
    data_df = pd.merge(features_df, labels_df, left_index=True, right_index=True)
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
    model = LogisticRegression(random_state=1)
    model.fit(X_train, y_train)
    return model



def train_sm(X_train, y_train):
    X_train = sm.add_constant(X_train)
    model = sm.Logit(y_train, X_train)
    model.fit()
    return model



def _get_model_id():
    return dt.datetime.now().strftime("%d%m%H%M")



def save_model(model, id):
    filepath = os.path.join(MODELS_DIR, f"model_{id}.joblib")
    joblib.dump(model, filepath)



def save_predictions(y_pred, id):
    filepath = os.path.join(PREDS_DIR, f"preds_{id}.json")
    with open(filepath, "w") as f:
        json.dump(y_pred, f, indent=4) # tmp



def main(args):

    data_df = load_x_y(args.features_filename, args.responses_filename)
    X_train, y_train, X_test, y_test = prepare_splits(data_df, 
                                                      test_size=0.2, 
                                                      random_state=1)

    ''' Training '''
    if args.lr_type == "sklearn":
        model = train(X_train, y_train)
    elif args.lr_type == "statsmodels":
        model = train_sm(X_train, y_train)

    ''' Model intepretation'''
    if args.lr_type == "sklearn":
        print(f"""Intercept: {model.intercept_}\n\nCoefficients: {model.coef_}\n\nScore: {model.score(X_train, y_train)}""")
    elif args.lr_type == "statsmodels":
        print(model.summary())
    
    ''' Evaluation '''
    if args.lr_type == "sklearn":
        y_pred = model.predict(X_test) # numpy array with numpy.bool_ values
        acc = accuracy_score(y_test, y_pred)
    elif args.lr_type == "statsmodels":
        y_pred = model.predict(X_test)
        print(y_pred)
        #y_pred = [True if x > 0.5 else False for x in y_pred]
    
    ''' Save model and predictions '''
    if args.save:
        id = _get_model_id()
        save_model(model, id)
        save_predictions(y_pred, id)





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
    
    parser.add_argument("--lr_type",
                        type=str,
                        default="sklearn",
                        help="Use LogisticRegression from sklearn or statsmodels")
    
    parser.add_argument("--save",
                        action="store_true",
                        help="include --save to save the model and predictions")
    
    args = parser.parse_args()
    main(args)




