
''' convert old responses in json format to jsonl format '''

import os
import json


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


responses_to_convert = 'responses/14081857_parsed_turbo_turbo.jsonl'


old_responses_path = os.path.join(PROJECT_DIR, responses_to_convert)
new_responses_path = old_responses_path.replace(".json", ".jsonl")

if responses_to_convert == 'responses/freetext_turbo_700dev_14081857.json':
    old_id_mapping_path = os.path.join(PROJECT_DIR, 'processed_uuids/uuids_14081857.json')

    # freetext
    with open(old_responses_path, "r") as f:
        responses = json.load(f)

    with open(old_id_mapping_path, "r") as f:
        id_mapping = json.load(f)

    with open(new_responses_path, "a+") as f:
        for idx, response in responses.items():
            new_response = {
                "idx": int(idx),
                "uuid": id_mapping[idx],
                "text": response
            }
            f.write(json.dumps(new_response) + "\n")


if responses_to_convert == 'responses/freetext_chatllama13_fulldev_03081353.json':
    # get id mapping from original data file
    data_path = os.path.join(PROJECT_DIR, 'data/commonsenseQA/dev_rand_split.jsonl')
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    with open(old_responses_path, "r") as f:
        responses = json.load(f)
    
    with open(new_responses_path, "a+") as f:
        for idx, response in responses.items():
            new_response = {
                "idx": int(idx),
                "uuid": data[int(idx)]['id'],
                "text": response
            }
            f.write(json.dumps(new_response) + "\n")
    

if responses_to_convert == 'responses/format_turbo_14081857_turbo.json':
    # new format: {idx:0, uuid:hiwhrih, text:…, format: {answer_letter: A, answer_text: bank}}

    old_id_mapping_path = os.path.join(PROJECT_DIR, 'processed_uuids/uuids_14081857.json')

    with open(old_id_mapping_path, "r") as f:
        id_mapping = json.load(f)
    
    with open(old_responses_path, "r") as f:
        responses = json.load(f)
    
    with open(new_responses_path, "a+") as f:
        for idx, response_dict in responses.items():
            new_response = {
                "idx": int(idx),
                "uuid": id_mapping[idx],
                "text": response_dict['full_text'],
                "format": {
                    "answer_letter": response_dict['answer_letter'],
                    "answer_text": response_dict['answer_text']
                }
            }
            if 'ERROR' in response_dict.keys():
                new_response['format']['ERROR'] = response_dict['ERROR']
            
            f.write(json.dumps(new_response) + "\n")


if responses_to_convert == 'responses/14081857_parsed_turbo_turbo.jsonl':
    # current format: {idx:0, uuid:hiwhrih, text:…, format: {answer_letter: A, answer_text: bank}}
    # only change format key to 'parsed'
    with open(old_responses_path, "r") as f:
        responses = [json.loads(line) for line in f.readlines()]
    
    new_responses_path = old_responses_path+'_new.jsonl'
    with open(new_responses_path, "a+") as f:
        for response in responses:
            response['parsed'] = response.pop('format')
            f.write(json.dumps(response) + "\n")