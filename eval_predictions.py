
import os
import json
import argparse

import utils.data



PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data/')
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses/')



def load_predictions(in_file):
    ''' Load predictions from jsonl file (*_parsed_*.jsonl)

    Format: {id: idx, uuid: uuid, response: freetext response, parsed: {answer_letter: X, answer_text: text}}, {...}

    prediction = answer_letter
    '''
    in_filepath = os.path.join(
        PROJECT_DIR, "responses", in_file
    ) if "/" not in in_file else os.path.join(PROJECT_DIR, in_file)

    with open(in_filepath, "r") as f:
        return [json.loads(line) for line in f.readlines()]



def write_predictions_eval(predictions_eval, in_file, mode="a+"):
    ''' Write evaluated predictions to jsonl file
    
    predictions_eval: list of dicts
    
    out format: json lines {id: idx, uuid: uuid, response: freetext response, format: {answer_letter: X, answer_text: text}, eval: {gold: gold, outcome: outcome}}, {...}
    '''
    out_file = f"{in_file.split('.')[0]}_eval.jsonl"
    out_filepath = os.path.join(
        PROJECT_DIR, "responses", out_file
    ) if "/" not in out_file else os.path.join(PROJECT_DIR, out_file)

    with open(out_filepath, mode) as f:
        for pred_with_eval in predictions_eval:
            f.write(json.dumps(pred_with_eval) + "\n")



def evaluate_CoQA(args):
    dump_size = 20
    predictions_eval = []

    data = utils.data.load_data(split=args.split, dataset='commonsenseQA')
    data = utils.data.flatten_CoQA_comprehension(data) # list of dicts
    predictions = load_predictions(args.in_file) # list of dicts

    # uuid:gold-label map for easy lookup
    uuid_label_map = {}
    for instance in data:
        uuid_label_map[instance['id']] = instance['answerKey']
        
    for idx, prediction in enumerate(predictions):

        try:
            instance_uuid = prediction['uuid']
            pred = prediction['parsed']['answer_letter']
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
                mode = "w+" if idx == dump_size else "a+"
                write_predictions_eval(predictions_eval, args.in_file, mode=mode)
                predictions_eval = []
        
        except Exception as error:
            print(f"Something went wrong for instance:\n{prediction}")
            print(f"\n{type(error).__name__} - {error}")

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_file",
                        type=str,
                        help="parsed responses file to evaluate: rel path | file name")
    parser.add_argument("--split",
                        type=str,
                        help="split to evaluate on: train | dev | test | merged")
    
    args = parser.parse_args()
    evaluate_CoQA(args)