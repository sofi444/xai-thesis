import os
import json
import argparse

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



def ensure_id_match(map_filename:str, predictions, data):
    map_path = os.path.join(PROJECT_DIR, "processed_uuids", map_filename)
    with open(map_path, "r") as f:
        map = json.load(f)
    
    for pred_idx in predictions.keys():
        # fetch corresponding uuid
        uuid = map[pred_idx]
        # fetch corresponding data
        if not uuid == data[int(pred_idx)]['id']:
            return False
        
    return True



def evaluate_CoQA(args):
    data = utils.data.load_data(split='dev', dataset='commonsenseQA', full_run=True)
    data = utils.data.flatten_CoQA_comprehension(data) # list
    predictions = load_predictions(args.predictions_filename) # dict

    if ensure_id_match(args.idmap_filename, predictions, data):
        
        predictions_with_outcome = {}

        for idx, pred_dict in predictions.items(): # idx is a string
            pred = pred_dict["answer_letter"]
            gold = data[int(idx)]["answerKey"]
            #print(pred, gold, pred_dict, data[int(idx)]["stem"])
            

            if pred == gold:
                pred_dict['outcome'] = True
            else: # catches wrong predictions and errors (ERROR:...)
                pred_dict['outcome'] = False

            predictions_with_outcome[idx] = pred_dict
        
        save_predictions_with_outcome(predictions_with_outcome, args)

    else:
        print("Error: ID mismatch between predictions and data.")
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions_filename",
                        type=str,
                        default="formatted_test.json")
    parser.add_argument("--idmap_filename",
                        type=str,
                        default="uuids_14081857.json")
    
    args = parser.parse_args()
    evaluate_CoQA(args)