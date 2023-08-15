
import json
import os
import argparse
import openai
import tqdm

import utils.output
import utils.models

from langchain import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI

from dotenv import load_dotenv, find_dotenv, dotenv_values



PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))



def load_freetext_responses(filename:str="test_json.json", full_run:bool=False):
    filepath = os.path.join(PROJECT_DIR, "responses", filename)
    
    with open(filepath, "r") as f:
        responses = json.load(f)
        
        if not full_run:
            N = 20 # tmp
            responses = {idx:res for idx, res in responses.items() if int(idx) < N}
            #interesting_cases = [14,17,53]
            #responses = {idx:res for idx,res in responses.items() if int(idx) in interesting_cases}
    
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

    template = PromptTemplate.from_template(load_template(template_version="v3"))

    output_parser, format_instructions = utils.output.build_parser(schemas_version="v3",
                                                                   parser_type="structured",
                                                                   only_json=False)
    
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
                        default="OpenAI-chat-default",
                        help="model to use")
    parser.add_argument("--in_filename",
                        type=str,
                        default="test_json.json",
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