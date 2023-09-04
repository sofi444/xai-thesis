
import json
import os
import argparse
import openai
import tqdm

import utils.output
import utils.models
import utils.codellama

from langchain import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI

from dotenv import load_dotenv, find_dotenv, dotenv_values



PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES_DIR = os.path.join(PROJECT_DIR, 'responses')



def load_freetext_responses(filename:str, full_run:bool=False):
    filepath = os.path.join(RESPONSES_DIR, filename)
    N = 7

    if filename.endswith(".json"):
        # Format: {idx: freetext response, ...}
        with open(filepath, "r") as f:
            responses = json.load(f)
            if not full_run:
                responses = {idx:res for idx, res in responses.items() if int(idx) < N}
    
    elif filename.endswith(".jsonl"):
        # Format: [{uuid: uuid, id: idx, response: freetext response}, ...]
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



def write_formatted_responses(formatted_responses:dict, out_filename:str):
    out_filepath = os.path.join(PROJECT_DIR, "responses", out_filename)
    with open(out_filepath, "w") as f:
        json.dump(formatted_responses, f, indent=4)



def load_template(template_version:str):
    template_filepath = os.path.join(PROJECT_DIR, 
                                     "prompt_templates/output-parsing_templates.json")
    with open(template_filepath, "r") as f:
        templates = json.load(f)
        template = templates[template_version]
    
    return template



def catch_errors(formatted_response:dict):
    ''' Add ERROR key to formatted_response if errors are found '''

    # multiple answers
    if len(formatted_response["answer_letter"]) > 1:
        formatted_response["ERROR"] = "multiple answers"
    if formatted_response["answer_letter"] == "":
        formatted_response["ERROR"] = "no answer"
    
    return formatted_response



def main(args):
    responses = load_freetext_responses(args.in_filename, args.full_run)

    if args.model == "openai_chat":
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
    
    elif args.model == "codellama":
        #model, tokenizer = utils.models.load_model(model=args.model)
        model, tokenizer = utils.codellama.load_codellama()

    template = PromptTemplate.from_template(load_template(template_version="v3"))

    output_parser, format_instructions = utils.output.build_parser(
        schemas_version="v3",
        parser_type="structured",
        only_json=False
    )
    
    if args.model == "codellama":
        inputs = []
        batch_size = 3
        formatted_responses = {}
        main_idx = 0

        instructions = utils.codellama.tokenize_instructions_and_examples(
            few_shot=True,
            tokenizer=tokenizer,
            instructions=format_instructions
        )

        for response in responses:
            if isinstance(response, dict): # batch responses jsonl
                inputs.append(response["response"])
            elif isinstance(response, str): # single response json
                inputs.append(response)
            
            if len(inputs) == batch_size or len(responses)-main_idx < batch_size:
                inputs = utils.codellama.tokenize_inputs(
                    inputs=inputs,
                    tokenizer=tokenizer
                )
                batch_inputs = utils.codellama.create_batch_inputs(
                    instructions_tok=instructions,
                    inputs_tok=inputs
                )
                batch_outputs = utils.codellama.run_batch(
                    model=model,
                    batch_encoded=batch_inputs
                )
                batch_outputs_decoded = utils.codellama.decode_batch(
                    batch_outputs=batch_outputs,
                    batch_inputs=batch_inputs,
                    tokenizer=tokenizer
                )
                
                batch_formatted = []
                for idx_batch, output in enumerate(batch_outputs_decoded):
                    try:
                        formatted = output_parser.parse(output) # dict
                        formatted = catch_errors(output) # dict
                    except:
                        formatted = {"ERROR": "unable to parse",
                                     "parser_output": output}
                    
                    if isinstance(response, dict): # batch responses jsonl
                        # add keys from original response
                        formatted['idx'] = response["idx"]
                        formatted['uuid'] = response["uuid"]
                        # add full text
                        formatted["full_text"] = response["response"]
                    
                    elif isinstance(response, str): # single response json
                        formatted["full_text"] = response
                        formatted["idx"] = main_idx

                    batch_formatted.append(formatted)
                    main_idx += 1

                inputs = [] # reset

                print(batch_formatted)
                print(main_idx)



    elif args.model == "openai":

        # structure of free text responses
        formatted_responses = {}
        for idx, response in tqdm.tqdm((responses.items())):
            prompt = template.format(freetext_response=response, 
                                    format_instructions=format_instructions)
            formatted_response = model.predict(prompt) # str

            try:
                formatted_response = output_parser.parse(formatted_response) # dict
                formatted_response = catch_errors(formatted_response)
            except:
                formatted_responses[idx] = {"ERROR": "unable to parse",
                                            "output_string": formatted_response,
                                            "full_text": response}
                continue

            # add full_text field to with original free text response
            if "full_text" not in formatted_response.keys():
                formatted_response["full_text"] = response

            formatted_responses[idx] = formatted_response

        write_formatted_responses(formatted_responses=formatted_responses,
                                out_filename=args.out_filename)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model",
                        type=str,
                        default="codellama",
                        help="model to use for parsing: openai_chat | codellama")
    parser.add_argument("--in_filename",
                        type=str,
                        default="freetext_turbo0301_700dev_14081857.json",
                        help="file with the free-text responses")
    parser.add_argument("--out_filename",
                        type=str,
                        default="formatted_responses.json",
                        help="file where to save the formatted responses")
    parser.add_argument("--use_azure",
                        action='store_true',
                        help="include to use OpenAI models from Azure, omit otherwise")
    parser.add_argument("--full_run",
                        action='store_true',
                        help="include to run on all instances from the datafile, omit to run on N instance (set N)")

    args = parser.parse_args()
    main(args)