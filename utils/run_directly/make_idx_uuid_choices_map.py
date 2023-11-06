
from datasets import load_from_disk
import json
import gzip
import os


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data/commonsenseQA')
SPLITS_DIR = os.path.join(PROJECT_DIR, 'classification/split_datasets/coqa')


data_path = os.path.join(DATA_DIR, "merged_filtered_10000.jsonl.gz")
hf_dataset = load_from_disk(SPLITS_DIR)


with gzip.open(data_path, 'rb') as f:
    data = [json.loads(line) for line in f]

map = {}

for i in range(len(hf_dataset['test'])):
    response_idx = int(hf_dataset['test'][i]['pandas_idx'])
    uuid = data[response_idx]['id']
    choices = data[response_idx]['question']['choices']

    choices_out = {
        choice['label']: choice['text'] for choice in choices
    }
    
    map[response_idx] = {'uuid': uuid, 'choices': choices_out}

with open('maps/idx_uuid_choices_map.json', 'w') as f:
    json.dump(map, f, indent=4)