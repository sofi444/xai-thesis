'''
{
    "answer": "This is the answer",
    "context": {
        "sentences": [["Sent 1"], ["Sent 2"]],
        "title": ["Title1", "Title 2"]
    },
    "id": "000001",
    "level": "hard",
    "question": "What is the answer?",
    "supporting_facts": {
        "sent_id": [0, 1, 3],
        "title": ["Title of para 1", "Title of para 2", "Title of para 3"]
    },
    "type": "bridge"
}

fullwiki
id: a string feature.
question: a string feature.
answer: a string feature.
type: a string feature.
level: a string feature.
supporting_facts: a dictionary feature containing:
title: a string feature.
sent_id: a int32 feature.
context: a dictionary feature containing:
title: a string feature.
sentences: a list of string features.

DatasetDict({
    train: Dataset({
        features: ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context'],
        num_rows: 90447
    })
    validation: Dataset({
        features: ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context'],
        num_rows: 7405
    })
    test: Dataset({
        features: ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context'],
        num_rows: 7405
    })
})
'''

import os
import json
import random
from datasets import load_dataset


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "data", "hotpotQA")


dataset = load_dataset("hotpot_qa", "fullwiki")

output_path = os.path.join(DATASET_DIR, "hotpotQA_10000.jsonl")

n_instances = 10000
n_easy = 3000
n_medium = 4000
n_hard = 3000

hotpotQA_10000 = []
# 3000 randomly sampled instances from val set (level = hard)
# 3000 randomly sampled instances from train set (level = easy)
# 4000 randomly sampled instances from train set (level = medium)


val_data = dataset['validation'].shuffle(seed=22).select(range(n_hard))
for idx, instance in enumerate(val_data):
    output = {
        "id": instance['id'],
        "question": instance['question'],
        "answer": instance['answer'],
        "type": instance['type'],
        "level": instance['level']
    }
    hotpotQA_10000.append(output)


train_data = dataset['train'].shuffle(seed=22)
for idx, instance in enumerate(train_data):
    level = instance['level']
    if level == 'easy' and n_easy > 0:
        n_easy -= 1
    elif level == 'medium' and n_medium > 0:
        n_medium -= 1
    else:
        continue
    
    output = {
        "id": instance['id'],
        "question": instance['question'],
        "answer": instance['answer'],
        "type": instance['type'],
        "level": instance['level']
    }
    hotpotQA_10000.append(output)

    if n_easy == 0 and n_medium == 0:
        break


random.shuffle(hotpotQA_10000)


with open(output_path, 'w') as f:
    for item in hotpotQA_10000:
        f.write(json.dumps(item) + "\n")