
''' convert old responses in json format to jsonl format '''

import os
import json


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


responses_to_convert = "responses/responses_12091031.jsonl" # change this


# convert responses files from json to jsonl
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


# changed key to parsed
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


# fix issue with wrong uuids
if '04091703' in responses_to_convert:
    # reload original data file to get the correct uuids
    data_path = os.path.join(PROJECT_DIR, 'data/commonsenseQA/train_filtered.jsonl')
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
        # only used the first 2000 for generation
    
    with open(old_responses_path, "r") as f:
        responses = [json.loads(line) for line in f.readlines()]
    
    new_responses_path = old_responses_path+'_new.jsonl'
    with open(new_responses_path, "a+") as f:
        for idx, response in enumerate(responses):
            response['uuid'] = data[idx]['id']
            f.write(json.dumps(response) + "\n")


# merge responses into one file with 10000 responses
if responses_to_convert == "responses/responses_12091031.jsonl":
    # 7300 responses in this file +
    # 2000 in 04091703 +
    # 700 in 14081857
    # = 10000 responses
    
    # indices order: 2000, 7300, 700

    filepath_7300 = os.path.join(PROJECT_DIR, responses_to_convert) # new generations
    filepath_2000 = os.path.join(PROJECT_DIR, "responses/04091703_freetext_turbo_2000train.jsonl")
    filepath_700 = os.path.join(PROJECT_DIR, "responses/14081857_freetext_turbo_700dev.jsonl")

    with open(filepath_7300, "r") as f:
        responses_7000 = [json.loads(line) for line in f.readlines()]
    with open(filepath_2000, "r") as f:
        responses_2000 = [json.loads(line) for line in f.readlines()]
    with open(filepath_700, "r") as f:
        responses_700 = [json.loads(line) for line in f.readlines()]
    
    merged_responses = [*responses_2000, *responses_7000, *responses_700]

    # out
    new_filepath = os.path.join(PROJECT_DIR, "responses", "12091031_freetext_turbo_10000.jsonl")
    main_idx = 0
    with open(new_filepath, "w+") as f:
        for response in merged_responses:
            response['idx'] = main_idx
            main_idx+=1
            f.write(json.dumps(response) + "\n")


# create a version of the original data which matches the 10000 responses
if responses_to_convert == "responses/responses_12091031.jsonl":

    train_filtered_path = os.path.join(PROJECT_DIR, "data/commonsenseQA/train_filtered.jsonl")
    dev_filtered_path = os.path.join(PROJECT_DIR, "data/commonsenseQA/dev_filtered.jsonl")
    
    with open(train_filtered_path, "r") as f:
        train_filtered = [json.loads(line) for line in f.readlines()]
    with open(dev_filtered_path, "r") as f:
        dev_filtered = [json.loads(line) for line in f.readlines()]
        dev_filtered = dev_filtered[:700]
    
    # 2000 responses are the first 2000 from train
    # 7300 responses are the remaining from train
    # 700 are the first 700 from dev
    merged_responses = [*train_filtered, *dev_filtered]

    new_filepath = os.path.join(
        PROJECT_DIR, 
        "data/commonsenseQA/merged_filtered_10000.jsonl"
    ) # would this give problems in data loading??
    with open(new_filepath, "w+") as f:
        for response in merged_responses:
            f.write(json.dumps(response) + "\n")

    
