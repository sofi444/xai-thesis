import os
import json
import gzip
import argparse
import tqdm

import utils.data



PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data/')
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses/')



def load_predictions(filename:str="formatted_test.json"):
    filepath = os.path.join(RESPONSES_DIR, filename)
    with open(filepath, "r") as f:
        predictions = json.load(f)
    return predictions



def save_predictions_with_outcome(predictions_with_outcome:dict, args):
    out_filename = f"{args.predictions_filename.split('.')[0]}_eval.json"
    out_path = os.path.join(RESPONSES_DIR, out_filename)
    with open(out_path, "w") as f:
        json.dump(predictions_with_outcome, f, indent=4)



def evaluate_CoQA(args):
    data = utils.data.load_data(split='dev', dataset='commonsenseQA', full_run=True)
    data = utils.data.flatten_CoQA_comprehension(data) # list
    predictions = load_predictions(args.predictions_filename) # dict
    predictions_with_outcome = {}

    for idx, pred_dict in tqdm.tqdm(predictions.items()): # idx is a string
        pred = pred_dict["prediction_letter"]
        gold = data[int(idx)]["answerKey"]
        #print(pred, gold, pred_dict, data[int(idx)]["stem"])

        if pred == gold:
            pred_dict['outcome'] = True
        else: # catches wrong predictions and errors (ERROR:...)
            pred_dict['outcome'] = False

        predictions_with_outcome[idx] = pred_dict
    
    save_predictions_with_outcome(predictions_with_outcome, args)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions_filename",
                        type=str,
                        default="formatted_test.json")
    
    args = parser.parse_args()
    evaluate_CoQA(args)