import os
import json
import pandas as pd
import datetime as dt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import statsmodels.api as sm

import joblib

import argparse



PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(PROJECT_DIR, "feature_extraction/featureExtraction/output/")
RESPONSES_DIR = os.path.join(PROJECT_DIR, "responses/")
MODELS_DIR = os.path.join(PROJECT_DIR, "classification/models/")
PREDS_DIR = os.path.join(PROJECT_DIR, "classification/preds/")
STATS_DIR = os.path.join(PROJECT_DIR, "classification/stats/")



def load_x_y(features_filename, responses_filename):
    features_df = pd.read_csv(
        os.path.join(FEATURES_DIR, features_filename))

    with open(os.path.join(RESPONSES_DIR, responses_filename), "r") as f:
        responses = json.load(f)

    idx_outcome_dict = {int(idx):res_dict['outcome'] for idx, res_dict in responses.items()}

    # tmp - only first 5 responses
    #idx_outcome_dict = {idx:outcome for idx, outcome in idx_outcome_dict.items() if idx < 5}

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



def train_statsmodels(X_train, y_train): # not in use
    X_train = sm.add_constant(X_train)
    model = sm.Logit(y_train, X_train)
    model.fit()
    return model



def _get_model_id():
    return dt.datetime.now().strftime("%d%m%H%M")



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

    
def save_stats(stats, id):

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    
    stats_filepath = os.path.join(STATS_DIR, f"stats_{id}.json")
    json_object = json.dumps(stats, cls=NumpyEncoder)

    with open(stats_filepath, "w") as f:
        f.write(json_object)



def main_sklearn(args):

    data_df = load_x_y(args.features_filename, args.responses_filename)
    X_train, y_train, X_test, y_test = prepare_splits(data_df, 
                                                      test_size=0.2, 
                                                      random_state=1)

    # idxs in test set
    test_idxs = X_test.index.tolist()

    ''' Training '''
    model = train(X_train, y_train)
    
    ''' Evaluation '''
    y_pred = model.predict(X_test) # numpy array with numpy.bool_ values
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nACCURACY: {accuracy}")

    ''' Model stats'''
    model_stats = {'intercept': model.intercept_, 
                   'coefficients': model.coef_,
                   'score': model.score(X_train, y_train),
                   'accuracy': accuracy}
    
    ''' Save model, predictions, stats '''
    if args.save:
        id = _get_model_id()
        save_model(model, id)
        save_stats(model_stats, id)
        save_predictions(y_pred, test_idxs, id)
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
    
    parser.add_argument("--lr_type",
                        type=str,
                        default="sklearn",
                        help="Use LogisticRegression from sklearn or statsmodels")
    
    parser.add_argument("--save",
                        action="store_true",
                        help="include --save to save the model and predictions")
    
    args = parser.parse_args()
    main_sklearn(args)




