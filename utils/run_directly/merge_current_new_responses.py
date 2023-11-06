
import os
import json
import datetime as dt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses')

# Load the current responses - force TSBS - id: 02111129
current_responses_path = os.path.join(RESPONSES_DIR, '02111129_parsed_turbo_10000_eval.jsonl')
with open(current_responses_path, 'r') as f:
    current_responses = [json.loads(line) for line in f]

# Load the new responses - force TSBS - id: 06111000
new_responses_path = os.path.join(RESPONSES_DIR, '06111000_parsed_turbo_454_eval.jsonl')
with open(new_responses_path, 'r') as f:
    new_responses = [json.loads(line) for line in f]

# Reindex the new responses (assign idx from 10000 onwards)
for response in new_responses:
    response['idx'] += 10000

# Merge the responses
all_responses = current_responses + new_responses

# Save the merged responses
merged_responses_path = os.path.join(
    RESPONSES_DIR, '06111210_parsed_turbo_10454_eval.jsonl'
)
with open(merged_responses_path, 'w') as f:
    for response in all_responses:
        f.write(json.dumps(response) + '\n')