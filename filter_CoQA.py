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
from tqdm import tqdm

import utils.data



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



def filter_data(data_split_to_filter, sim_threshold:float, gramm_threshold:float):
    ''' Filters commonsenseQA data based on three conditions '''

    data = utils.data.load_data(split=data_split_to_filter,
                                dataset="commonsenseQA",
                                filtered=False,
                                num_instances=None) # None for all instances else int

    print(f"Loaded {len(data)} instances from original {data_split_to_filter} data.")

    filtered_data = []
    for instance in tqdm(data):
        question = instance['question']['stem']
        choices = instance['question']['choices']
    
        # first condition
        if check_choice_uniqueness(choices):
            # second condition
            if check_semantic_similarity(choices, sim_threshold, nlp):
                
                #third condition
                #lm_score = get_lm_score(question, model, tokenizer)
                #if lm_score <= gramm_threshold:
                    
                #    filtered_data.append(instance)
                #else:
                #    print(f"!cond 3 (grammaticality) - {question} - {lm_score}\n")

                filtered_data.append(instance)
            else:
                print(f"!cond 2 (semantic similarity) - {question} - {choices}\n")
        else:
            print(f"!cond 1 (choice uniqueness) - {question} - {choices}\n")

    return filtered_data


def save_filtered_data(filtered_data, data_split_to_filter):
    ''' Saves filtered data to jsonl file 
    out filename: <split>_filtered.jsonl
    '''
    out_path = os.path.join(DATASET_DIR, f"{data_split_to_filter}_filtered.jsonl")
    with open(out_path, 'w') as f:
        for instance in filtered_data:
            json.dump(instance, f)
            f.write('\n')
    
    print("Filtered data saved to file.")



if __name__ == "__main__":

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) # xai_sofia_casadei_master_thesis
    DATA_DIR = os.path.join(PROJECT_DIR, 'data/') # xai_sofia_casadei_master_thesis/data/
    DATASET_DIR = os.path.join(DATA_DIR, 'commonsenseQA')

    sim_threshold = 0.88
    gramm_threshold = 80.00

    filtered_data = filter_data(data_split_to_filter="train", 
                                sim_threshold=sim_threshold,
                                gramm_threshold=gramm_threshold)
    
    print(f"Instances in filtered data: {len(filtered_data)}")

    save_filtered_data(data_split_to_filter="train", filtered_data=filtered_data)