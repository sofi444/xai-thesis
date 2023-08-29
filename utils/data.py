
import os
import gzip
import json
import logging
import time
from typing import List, Dict



PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # xai_sofia_casadei_master_thesis
DATA_DIR = os.path.join(PROJECT_DIR, 'data/') # xai_sofia_casadei_master_thesis/data/



def load_data(split:str='dev', dataset:str='commonsenseQA', num_instances:int=None, 
              filtered:bool=True) -> List[Dict]:

    DATASET_DIR = os.path.join(DATA_DIR, dataset)
    
    if split not in ['train', 'dev', 'test']:
        raise ValueError("split arg must be 'train', 'dev' or 'test'")
    
    files_from_split = [f for f in os.listdir(DATASET_DIR) if split in f]

    if filtered:
        filenames = [f for f in files_from_split if 'filtered' in f]
    else:
        filenames = files_from_split
    
    if len(filenames) == 1:
        filename = filenames[0]
    elif len(filenames) > 1:
        # if both json and gz versions exist, load from gz
        filename = [f for f in filenames if '.gz' in f][0]

    if not filename:
        raise FileNotFoundError(
            f"Split: {split}, Filtered: {filtered}, Dir: {DATASET_DIR}"
            )
    filepath = os.path.join(DATASET_DIR, filename)

    print(f"Loading {num_instances if num_instances else 'all'} instances from {filepath}")
    
    if filename.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f:
            if num_instances:
                return [json.loads(line) for i, line in enumerate(f) if i < num_instances]
            else:
                return [json.loads(line) for line in f]

    elif filename.endswith('.jsonl'):
        with open(filepath, 'r') as f:
            if num_instances:
                return [json.loads(line) for i, line in enumerate(f) if i < num_instances]
            else:
                return [json.loads(line) for line in f]



def flatten_CoQA(data: List[Dict]) -> List[Dict]:
    ''' Flatten instances of CoQA from nested dicts to one-level dicts '''
    start_time = time.time()
    data_flat = []

    for instance in data:
        instance_flat = {}
        for k,v in instance.items():
            if isinstance(v, dict): # question
                for inner_k, inner_v in v.items():
                    if isinstance(inner_v, list): # choices
                        for choice in inner_v:
                            letter = choice['label']
                            instance_flat[f"choice_{letter}"] = choice['text']
                    else:
                        instance_flat[inner_k] = inner_v
            else:
                instance_flat[k] = v
                
        data_flat.append(instance_flat)
    
    end_time = time.time()
    #print(f"flatten_CoQA(): {end_time - start_time} seconds")

    return data_flat



def flatten_CoQA_comprehension(data: List[Dict]) -> List[Dict]:
    ''' Flatten instances of CoQA from nested dicts to one-level dicts
        using list and dict comprehensions '''
    
    start_time = time.time()

    data_flat = [
        {
            inner_k if not isinstance(v, dict) else f"choice_{choice['label']}" if isinstance(inner_v, list) else inner_k: inner_v if not isinstance(v, dict) else choice['text'] if isinstance(inner_v, list) else inner_v
            for k, v in instance.items()
            for inner_k, inner_v in (v.items() if isinstance(v, dict) else [(k, v)])
            for choice in (inner_v if isinstance(inner_v, list) else [None])
        }
        for instance in data
    ]

    end_time = time.time()
    #print(f"flatten_CoQA_comprehension(): {end_time - start_time} seconds")

    return data_flat




if __name__ == "__main__":

    load_data(split='dev', num_instances=10)