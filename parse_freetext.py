
import json
import os
import argparse
import openai
import torch

import utils.output
import utils.models
import utils.codellama
import utils.prompting
import utils.data

from langchain import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI

from dotenv import load_dotenv, find_dotenv, dotenv_values

from tqdm import tqdm


'''
device('cuda') uses all gpus, set CUDA_VISIBLE_DEVICES before running script
    CUDA_VISIBLE_DEVICES=0 python3 parse_freetext.py

setting gpus as script ('cuda:0') give invald string error
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')



def load_freetext_responses(in_file:str, full_run:bool=False):
    ''' 
    Load freetext from jsonl file
    Format: {id: idx, uuid: uuid, response: freetext response}, {...}
    '''
    filepath = os.path.join(
        PROJECT_DIR, "responses", in_file
    ) if "/" not in in_file else os.path.join(PROJECT_DIR, in_file)

    N = 15
    with open(filepath, "r") as f:
        responses = [json.loads(line) for line in f.readlines()]
        if not full_run:
            responses = responses[:N]
    
    return responses



def config_env():
    _ = load_dotenv(find_dotenv())
    if _ is not True:
        raise Exception("No .env file found")
    return dict(dotenv_values('.env'))



def write_parsed_responses(parsed_responses, in_file):
    ''' Write parsed responses to jsonl file
    parsed_responses: list of dicts
    out parsed: json lines {id: idx, uuid: uuid, response: freetext response, parsed: {answer_letter: X, answer_text: text}}, {...}
    '''
    out_file = in_file.replace("freetext", "parsed")
    if in_file.endswith(".gz"):
        out_file = out_file.replace(".gz", "")
    
    out_filepath = os.path.join(
        PROJECT_DIR, "responses", out_file
    ) if "/" not in out_file else os.path.join(PROJECT_DIR, out_file)

    if not out_filepath.endswith(".jsonl"):
        out_filepath += ".jsonl"
        
    with open(out_filepath, "a+") as f:
        for response in parsed_responses:
            f.write(json.dumps(response) + "\n")



def load_template(template_version:str):
    template_filepath = os.path.join(PROJECT_DIR, 
                                     "prompt_templates/output-parsing_templates.json")
    with open(template_filepath, "r") as f:
        templates = json.load(f)
        template = templates[template_version]
    
    return template



def catch_errors(parsed_response:dict):
    ''' Add ERROR key to parsed_response if errors are found '''
    
    if len(parsed_response["answer_letter"]) > 1:
        parsed_response["ERROR"] = "multiple answers"
    if parsed_response["answer_letter"] == "":
        parsed_response["ERROR"] = "no answer"
    
    return parsed_response



def get_uuid_choices_map(split_to_load, responses):

    data = utils.data.load_data(
        split=split_to_load,
        dataset="commonsenseQA",
        num_instances=None,
        filtered=True
    )
    
    uuid_choices_map = {}
    for instance in data:
        choices = instance["question"]["choices"]
        choices = ", ".join([f"{choice['label']}. {choice['text']}" for choice in choices])
        uuid_choices_map[instance["id"]] = choices     
    
    return uuid_choices_map



def main(args):
    responses = load_freetext_responses(args.in_file, args.full_run)

    template = PromptTemplate.from_template(load_template(template_version="v3"))

    output_parser, parsed_instructions = utils.output.build_parser(
        schemas_version="v3",
        parser_type="structured",
        only_json=False
    )

    if args.with_choices:
        uuid_choices_map = get_uuid_choices_map(
            split_to_load=args.og_data_split, 
            responses=responses
        )

    if args.model == "codellama":
        #model, tokenizer = utils.models.load_model(model=args.model)
        model, tokenizer = utils.codellama.load_codellama()

        sys_prompt = template.template.split("\n\n")[0] + "\n\n" + parsed_instructions
        # if few shot, append examples to sys prompt

        parsed_responses = []
        dump_size = 16
        main_idx = 0

        for response in tqdm(responses):
            # sys prompt stays the same for all
            user_message = response["text"]

            if args.with_choices:
                choices = uuid_choices_map[response["uuid"]]
                user_message = f"{choices}\n\n{user_message}"

            llama_prompt = utils.prompting.get_llama_prompt(
                sys_prompt=sys_prompt,
                user_message=user_message,
            )

            inputs = tokenizer(
                llama_prompt,
                return_tensors="pt",
            ).to(device)

            output = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                temperature=0.1,
                max_new_tokens=70,
                do_sample=False,
                top_p=0.5,
            )[0].to('cpu')

            output_decoded = tokenizer.decode(
                output[inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            try:
                parsed = output_parser.parse(output_decoded) # dict
                parsed = catch_errors(parsed) # dict
            except:
                parsed = {
                    "ERROR": "unable to parse", 
                    "parser_output": output_decoded
                } # change
            
            # combine response and parsed dicts
            output_dict = {**response, 'parsed': parsed}
        
            parsed_responses.append(output_dict)
            main_idx += 1

            if len(parsed_responses) == dump_size or len(responses)-main_idx < dump_size:
                write_parsed_responses(
                    parsed_responses=parsed_responses,
                    in_file=args.in_file
                )
                parsed_responses = []
            


    elif args.model == "openai_chat":
        env_dict = config_env()
        if args.use_azure:
            openai.api_type = "azure"
            openai.api_base = env_dict["AZURE_OPENAI_ENDPOINT"]
            openai.api_version = env_dict["OPENAI_DEPLOYMENT_VERSION"]
            openai.api_key = env_dict["AZURE_OPENAI_KEY"]
            openai.proxy = env_dict["HTTP_PROXY"]

        model = utils.models.load_model(env_dict=env_dict,
                                        use_azure=args.use_azure,
                                        model=args.model)
        
        inputs = []
        batch_size = 4
        main_idx = 0

        for response in responses:
            # response is dict (line from jsonl)
            prompt = template.parsed(freetext_response=response["text"],
                                     parsed_instructions=parsed_instructions)
            inputs.append(prompt)

            if len(inputs) == batch_size or len(responses)-main_idx < batch_size:

                batch_outputs = model.batch(inputs) # call

                batch_parsed = []
                for output in batch_outputs:
                    try:
                        parsed = output_parser.parse(output) # dict
                        parsed = catch_errors(output) # dict
                    except:
                        parsed = {
                            "ERROR": "unable to parse",
                            "parser_output": output
                        }
                    
                    # combine response and parsed dicts
                    output_dict = {**response, 'parsed': parsed}
                    
                    batch_parsed.append(output_dict)
                    main_idx += 1

                inputs = [] # reset

                write_parsed_responses(
                    parsed_responses=batch_parsed,
                    in_file=args.in_file
                )





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model",
                        type=str,
                        default="codellama",
                        help="model to use for parsing: openai_chat | codellama")
    parser.add_argument("--in_file",
                        type=str,
                        default="freetext_turbo_700dev_14081857.jsonl",
                        help="file with the free-text responses: rel path | file name")
    parser.add_argument("--use_azure",
                        action='store_true',
                        help="include to use OpenAI models from Azure, omit otherwise")
    parser.add_argument("--full_run",
                        action='store_true',
                        help="include to run on all instances from the datafile, omit to run on N instance (set N)")
    parser.add_argument("--with_choices",
                        action='store_true',
                        help="include to add choices to the prompt, omit otherwise.")
    parser.add_argument("--og_data_split",
                        type=str,
                        help="split of original data to get choices from: train | dev | test | merged")
    

    args = parser.parse_args()
    main(args)