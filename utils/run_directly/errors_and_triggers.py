import os
import gzip
import json


write_errors = False
search_errors = True
find_triggers = True
write_triggers = True


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses/')
DATA_DIR = os.path.join(PROJECT_DIR, 'data/commonsenseQA/')

responses_path = os.path.join(RESPONSES_DIR, "12091031_parsed_turbo_10000_eval.jsonl")
data_path = os.path.join(DATA_DIR, "merged_filtered_10000.jsonl.gz")

errors_path = os.path.join(RESPONSES_DIR, "12091031_errors.jsonl")
triggers_path = os.path.join(DATA_DIR, "merged_filtered_10000_triggers.jsonl")


with open(responses_path) as f:
    responses = [json.loads(line) for line in f]


if search_errors:
    no_answer = []
    multiple_answers = []

    for response in responses:
        if 'ERROR' in response['parsed'].keys():
            if response['parsed']['ERROR'] == 'multiple answers':
                multiple_answers.append(response)
            elif response['parsed']['ERROR'] == 'no answer':
                no_answer.append(response)
            else:
                print(response)


if write_errors:
    with open(errors_path, 'w') as f:
        for response in (no_answer + multiple_answers):
            f.write(json.dumps(response) + '\n')


if find_triggers:
    with gzip.open(data_path) as f:
        data = [json.loads(line) for line in f]

    errors_ids = [response['uuid'] for response in (no_answer + multiple_answers)]

    triggers = []
    for instance in data:
        instance_id = instance['id']
        if instance_id in errors_ids:
            triggers.append(instance)
    
    print(len(triggers))
    print(len(errors_ids))


if write_triggers:

    with open(triggers_path, 'w') as f:
        for trigger in triggers:
            f.write(json.dumps(trigger) + '\n')