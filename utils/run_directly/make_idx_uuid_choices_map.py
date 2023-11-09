
from datasets import load_from_disk
import json
import gzip
import os
import sys


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data/commonsenseQA')
SPLITS_DIR = os.path.join(PROJECT_DIR, 'classification/split_datasets')
MAPS_DIR = os.path.join(PROJECT_DIR, 'maps')

sys.path.append(PROJECT_DIR)
import utils.data

splits_to_use = 'coqa_force_aug' # 'coqa_force_aug', 'coqa_force', 'coqa'
splits_to_use_path = os.path.join(SPLITS_DIR, splits_to_use)
hf_dataset = load_from_disk(splits_to_use_path)

#data_to_use = ['merged_filtered_10000.jsonl.gz'] coqa, coqa force
data_to_use = ['merged_filtered_10000.jsonl.gz', 'new_instances_filtered.jsonl'] # coqa force aug
data_to_use = [os.path.join(DATA_DIR, data) for data in data_to_use]

if len(data_to_use) == 1:
    data = utils.data.load_data(split='merged')
else:
    data_1 = utils.data.load_data(split='merged')
    data_2 = utils.data.load_data(split='new')
    data = data_1 + data_2

map = {}

for i in range(len(hf_dataset['test'])):
    response_idx = int(hf_dataset['test'][i]['pandas_idx'])
    uuid = data[response_idx]['id']
    choices = data[response_idx]['question']['choices']

    choices_out = {
        choice['label']: choice['text'] for choice in choices
    }
    
    map[response_idx] = {'uuid': uuid, 'choices': choices_out}


output_map_name = f'idx_uuid_choices_map_{splits_to_use}.json'
output_map_path = os.path.join(MAPS_DIR, output_map_name)
with open(output_map_path, 'w') as f:
    json.dump(map, f, indent=4)