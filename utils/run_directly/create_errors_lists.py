
import os
import json


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses')
MAPS_DIR = os.path.join(PROJECT_DIR, 'maps')


# Load parsed responses
responses_path_10000 = os.path.join(RESPONSES_DIR, '02111129_parsed_turbo_10000.jsonl')
responses_path_new = os.path.join(RESPONSES_DIR, '06111000_parsed_turbo_454.jsonl')

with open(responses_path_10000, 'r') as f:
    responses_10000 = [json.loads(line) for line in f.readlines()]

with open(responses_path_new, 'r') as f:
    responses_new = [json.loads(line) for line in f.readlines()]

# Re-index new responses (assing idx from 10000 onwards)
for response in responses_new:
    response['idx'] = response['idx'] + 10000

# Find errors
errors_idx_uuid_map = {}
for response in responses_10000+responses_new:
    if 'ERROR' in response['parsed'].keys():
        errors_idx_uuid_map[response['idx']] = response['uuid']

# Save errors
errors_path = os.path.join(MAPS_DIR, 'errors_idx_uuid_map.json')
with open(errors_path, 'w') as f:
    json.dump(errors_idx_uuid_map, f, indent=4)



