
import os
import json
import argparse

import utils.data



PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data/')
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses/')



def load_predictions(filename):
    ''' Load predictions from jsonl file (format_*.json)

    Format: {id: idx, uuid: uuid, response: freetext response, format: {answer_letter: X, answer_text: text}}, {...}

    prediction = answer_letter
    '''
    filepath = os.path.join(RESPONSES_DIR, filename)

    with open(filepath, "r") as f:
        return [json.loads(line) for line in f.readlines()]



def write_predictions_eval(predictions_eval, args):
    ''' Write evaluated predictions to jsonl file
    
    predictions_eval: list of dicts
    
    out format: json lines {id: idx, uuid: uuid, response: freetext response, format: {answer_letter: X, answer_text: text}, eval: {gold: gold, outcome: outcome}}, {...}
    '''
    out_filename = f"{args.in_filename.split('.')[0]}_eval.jsonl"
    out_filepath = os.path.join(RESPONSES_DIR, out_filename)

    with open(out_filepath, "a+") as f:
        for pred_with_eval in predictions_eval:
            f.write(json.dumps(pred_with_eval) + "\n")



def evaluate_CoQA(args):
    dump_size = 10
    predictions_eval = []

    data = utils.data.load_data(split='dev', dataset='commonsenseQA', full_run=True)
    data = utils.data.flatten_CoQA_comprehension(data) # list of dicts
    predictions = load_predictions(args.in_filename) # list of dicts

    # uuid:gold-label map for easy lookup
    uuid_label_map = {}
    for instance in data:
        uuid_label_map[instance['id']] = instance['answerKey']
        
    for idx, prediction in enumerate(predictions):
        instance_uuid = prediction['uuid']
        pred = prediction['format']['answer_letter']
        gold = uuid_label_map[instance_uuid]

        eval_dict = {'gold': gold, 'outcome': None}
        if pred == gold:
            eval_dict['outcome'] = True
        else:
            eval_dict['outcome'] = False
        
        prediction['eval'] = eval_dict
        predictions_eval.append(prediction)

        # write to file every <dump_size> instances
        if idx == dump_size or len(predictions)-idx < dump_size:
            write_predictions_eval(predictions_eval, args)
            predictions_eval = []
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_filename",
                        type=str,
                        default="formatted_test.json")
    
    args = parser.parse_args()
    evaluate_CoQA(args)