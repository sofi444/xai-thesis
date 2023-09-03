
import os   
import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain import PromptTemplate

import prompting
import output



gpus_to_use = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
device = torch.device("cuda:"+gpus_to_use if torch.cuda.is_available() else "cpu")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_DIR, '.cache')
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses')

schemas_path = os.path.join(PROJECT_DIR, 'schemas/output-parsing_schemas.json')
template_path = os.path.join(PROJECT_DIR, 'prompt_templates/output-parsing_templates.json')
responses_path = os.path.join(RESPONSES_DIR, 'freetext_turbo0301_700dev_14081857.json')



def load_codellama():
    model_id = "codellama/CodeLlama-7b-Instruct-hf" # 7b, 13b, 34b
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        #device_map='auto',
        cache_dir=CACHE_DIR,
    )
    model.to(device)
    print(model.device)
    print(device)

    return model, tokenizer

'''

template_str = json.load(open(template_path, "r"))['v3']
template = PromptTemplate.from_template(template_str)

output_parser, format_instructions = output.build_parser(schemas_version="v3",
                                                         parser_type="structured",
                                                         only_json=False)

responses = json.load(open(responses_path, "r"))
responses = {int(idx): response.strip('\n') for idx, response in responses.items()}

'''

if __name__ == "__main__":
    load_codellama()