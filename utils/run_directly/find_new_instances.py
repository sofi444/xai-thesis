
import os
import json
import gzip


# Directories
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses')
DATA_DIR = os.path.join(PROJECT_DIR, 'data/commonsenseQA')


# Load data
train_path = os.path.join(DATA_DIR, 'train_filtered.jsonl.gz')
dev_path = os.path.join(DATA_DIR, 'dev_filtered.jsonl')

with gzip.open(train_path, "rt") as f:
    train_data = [json.loads(line) for line in f.readlines()]

with open(dev_path, "r") as f:
    dev_data = [json.loads(line) for line in f.readlines()]

# Load responses - 10000
responses_path = os.path.join(RESPONSES_DIR, '02111129_freetext_turbo_10000.jsonl')

with open(responses_path, "r") as f:
    responses_data = [json.loads(line) for line in f.readlines()]


# Find ids in tran or dev that are not in responses
responses_ids = [response['uuid'] for response in responses_data]

train_ids = [train['id'] for train in train_data]
dev_ids = [dev['id'] for dev in dev_data]

new_ids = []
for id in train_ids+dev_ids:
    if id not in responses_ids:
        new_ids.append(id)

#print(len(new_ids)) # 454

# Find new instances
new_instances = []
for instance in train_data+dev_data:
    if instance['id'] in new_ids:
        new_instances.append(instance)

new_instances_path = os.path.join(DATA_DIR, 'new_instances.jsonl')

with open(new_instances_path, 'w') as f:
    for instance in new_instances:
        json.dump(instance, f)
        f.write('\n')