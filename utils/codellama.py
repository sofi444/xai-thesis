
import os   
import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.tokenization_utils_base import BatchEncoding

from langchain import PromptTemplate



gpus_to_use = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
device = torch.device(f"cuda:{gpus_to_use}" if torch.cuda.is_available() else "cpu")
# ! does not work with multiple gpus: invalid device string

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_DIR, '.cache')
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses')

schemas_path = os.path.join(PROJECT_DIR, 'schemas/output-parsing_schemas.json')
template_path = os.path.join(PROJECT_DIR, 'prompt_templates/output-parsing_templates.json')
responses_path = os.path.join(RESPONSES_DIR, 'freetext_turbo0301_700dev_14081857.json')
examples_path = os.path.join(PROJECT_DIR, 'output-parsing_examples.jsonl')



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

    print(f"Loaded model on device: {device} | {model.device}")
    return model, tokenizer



def tokenize_instructions_and_examples(few_shot:bool, tokenizer, instructions):

    if few_shot:
        N_examples = 6
        raw_examples = []
        with open(examples_path, "r") as f:
            for line in f:
                raw_examples.append(json.loads(line))

        examples = "\n\nExamples:\n"
        for example in raw_examples[:N_examples]:
            examples += example['full_text'].replace('\n', ' ')
            examples += f"\n\n```json\n{example['output']}\n```\n\n"

        instructions += examples

    instructions_tokenized = tokenizer(instructions, return_tensors="pt").to(device)

    return instructions_tokenized



def tokenize_inputs(inputs, tokenizer):
    tokenized_inputs = []
    for input in inputs:
        tokenized_inputs.append(
            tokenizer(input, return_tensors="pt").to(device)
    )
    return tokenized_inputs


def create_batch_inputs(instructions_tok, inputs_tok):
    
    combined_input_ids = [torch.cat(
        [instructions_tok['input_ids'],
        input_tok['input_ids']],
        dim=1) for input_tok in inputs_tok
    ]

    combined_attention_mask = [torch.cat(
        [instructions_tok['attention_mask'],
        input_tok['attention_mask']],
        dim=1) for input_tok in inputs_tok
    ]

    # pad all tensors to have same length
    max_len = max([x.squeeze().numel() for x in combined_input_ids])

    padded_input_ids = [torch.nn.functional.pad(
        tensor, 
        pad=(0, max_len - tensor.numel()), 
        mode='constant', 
        value=0
    ) for tensor in combined_input_ids]

    padded_attention_mask = [torch.nn.functional.pad(
        tensor,
        pad=(0, max_len - tensor.numel()),
        mode='constant',
        value=0
    ) for tensor in combined_attention_mask]

    batch_encoded = BatchEncoding(
        {'input_ids': torch.stack(padded_input_ids).squeeze(1),
         'attention_mask': torch.stack(padded_attention_mask).squeeze(1)}
        )
    
    return batch_encoded

    

def run_batch(model, batch_encoded):
    return model.generate(
        **batch_encoded,
        do_sample=False,
        max_new_tokens=50,
        top_p=0.5,
        temperature=0.1,
    )


def decode_batch(batch_outputs, batch_inputs, tokenizer):
    return tokenizer.batch_decode(
        batch_outputs[:, batch_inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )



if __name__ == "__main__":
    load_codellama()