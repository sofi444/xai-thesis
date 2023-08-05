"""
3rd condition does not work. Change method for cheking grammatical correctness.
"""

import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import spacy
import en_core_web_sm
from scipy.spatial.distance import cosine

from itertools import combinations
from typing import List, Dict



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.load('en_core_web_sm') # model for semantic similarity

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased').to(device)



def check_choice_uniqueness(choices: List[Dict]) -> bool:
    ''' Returns True if no two choices are the same, False otherwise '''

    texts = [choice['text'] for choice in choices]
    
    return len(texts) == len(set(texts))



def check_semantic_similarity(choices, sim_threshold, nlp):
    ''' 
    Returns True if all choices are semantically different (based on threshold),
    False otherwise.
    '''

    texts = [choice['text'] for choice in choices]
    vectors = [nlp(text).vector for text in texts]
        
    similarities = [1 - cosine(v1, v2) for v1, v2 in combinations(vectors, 2)]

    return all(similarity <= sim_threshold for similarity in similarities)



def get_lm_score(question, model, tokenizer):
    inputs = tokenizer(question, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    perplexity = torch.exp(loss)

    return round(perplexity.item(), 2)



def filter_data(file_path:str, sim_threshold:float, gramm_threshold:float):

    filtered_data = []

    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            choices = data['question']['choices']
            question = data['question']['stem']

            # first condition
            if check_choice_uniqueness(choices):
                # second condition
                if check_semantic_similarity(choices, sim_threshold, nlp):
                    
                    #third condition
                    #lm_score = get_lm_score(question, model, tokenizer)
                    #if lm_score <= gramm_threshold:
                    #    filtered_data.append(data)

                    #else:
                    #    print(f"{idx+1} !cond 3 (grammaticality) - {question} - {lm_score}\n")

                    filtered_data.append(data)
                else:
                    print(f"{idx+1} !cond 2 (semantic similarity) - {question} - {choices}\n")
            else:
                print(f"{idx+1} !cond 1 (choice uniqueness) - {question}")
            
            #if idx == 200: #tmp
            #    break

    return filtered_data



def save_filtered_data(filtered_data, file_path):
    ''' Saves filtered data to file_path '''

    with open(file_path, 'w') as f:
        for data in filtered_data:
            json.dump(data, f)
            f.write('\n')



if __name__ == "__main__":

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) # xai_sofia_casadei_master_thesis
    DATA_DIR = os.path.join(PROJECT_DIR, 'data/') # xai_sofia_casadei_master_thesis/data/
    DATASET_DIR = os.path.join(DATA_DIR, 'commonsenseQA')
    
    file_path = os.path.join(DATASET_DIR, 'dev_rand_split.jsonl')

    sim_threshold = 0.88
    gramm_threshold = 60.00

    filtered_data = filter_data(file_path, 
                                sim_threshold, 
                                gramm_threshold)
    
    print(f"Instances in filtered data: {len(filtered_data)}")
    print(f"Instances in original data: {len(open(file_path, 'r').readlines())}")

    save_filtered_data(filtered_data, os.path.join(DATASET_DIR, 'dev_rand_split_filtered.jsonl'))
    print("Filtered data saved to file.")