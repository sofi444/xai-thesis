
from datasets import load_from_disk
import json
import gzip


splits_path = "/mount/studenten-temp1/users/dpgo/xai-thesis/classification/split_datasets/coqa"
data_path = "/mount/studenten-temp1/users/dpgo/xai-thesis/data/commonsenseQA/merged_filtered_10000.jsonl.gz"

hf_dataset = load_from_disk(splits_path)

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