
import argparse
import pprint as pp
import openai
import os
import tqdm
import json

from dotenv import load_dotenv, find_dotenv, dotenv_values

import utils.data
import utils.models
import utils.prompting
import utils.output



def config_env():
    _ = load_dotenv(find_dotenv())
    if _ is not True:
        raise Exception("No .env file found")
    
    return dict(dotenv_values('.env'))



def get_openai_attributes():
    return {
        att:getattr(openai, att) for att in dir(openai) if att in [
            "api_type", "api_version", "api_base", "api_key", "api_proxy"
            ]
        }



def main(args):
    print(f"Arguments:\n{pp.pformat(vars(args))}\n")

    # env: check + get values
    env_dict = config_env()
    if args.use_azure:
        openai.api_type = "azure"
        openai.api_base = env_dict["AZURE_OPENAI_ENDPOINT"]
        openai.api_version = env_dict["OPENAI_DEPLOYMENT_VERSION"]
        openai.api_key = env_dict["AZURE_OPENAI_KEY"]
        openai.proxy = env_dict["HTTP_PROXY"]

    # load data
    data = utils.data.flatten_CoQA_comprehension(
        utils.data.load_data(
            split=args.data_split,
            filtered=True,
            num_instances=10 if not args.full_run else None)
        )
    print(f"\tLoaded {len(data)} instances")

    # create prompt
    prompt_template = utils.prompting.create_template(
        dataset="commonsenseQA",
        prompting_type=args.prompting_type,
        output_formatting=args.output_formatting,
        for_llama2=args.llama_prompt)

    if args.output_formatting:
        output_parser, format_instructions = utils.output.get_format_instructions(
            schemas_src="default", 
            parser_type="structured", 
            only_json=False)

    # load model
    model = utils.models.load_model(model=args.model,
                                    use_azure=args.use_azure,
                                    env_dict=env_dict)

    # create chain
    chain = utils.models.create_LLMchain(llm=model,
                                         prompt_template=prompt_template,
                                         verbose=args.chain_verbose)

    # pass inputs and run chain
    all_responses = {}
    idx_uuid_map = {}
    inputs = []
    batch_size = 4
    run_id = utils.output.get_run_id()
    main_idx = 0

    for idx, instance in tqdm.tqdm(enumerate(data)):
        idx_uuid_map[idx] = instance["id"]
        inputs.append({
            'question':instance["stem"],
            'choice_A':instance["choice_A"],
            'choice_B':instance["choice_B"],
            'choice_C':instance["choice_C"],
            'choice_D':instance["choice_D"],
            'choice_E':instance["choice_E"]
            })
        
        if len(inputs) == batch_size or len(data)-idx < batch_size:
            batch_responses = chain.batch(inputs) # call
            inputs = [] # reset inputs
            batch_responses_out = []

            for response in batch_responses:
                batch_responses_out.append(
                    {'uuid':idx_uuid_map[main_idx],
                     'idx':main_idx,
                     'response':response['text']}
                )
                main_idx += 1
                
                all_responses[main_idx] = response['text'] # save all responses in one dict

            if args.save:
                utils.output.write_batch_responses(batch_responses_out, run_id)

    if args.save:
        utils.output.write_all_responses(all_responses, run_id, idx_uuid_map)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_split", 
                        type=str, default="dev", 
                        help="data split to use")
    parser.add_argument("--model", 
                        type=str, 
                        default="openai_chat", 
                        help="model to use: openai_chat | llama2_chat | llama2_chat_local")
    parser.add_argument("--use_azure", 
                        action='store_true', 
                        help="include to use OpenAI models from Azure, omit otherwise")
    parser.add_argument("--full_run",
                        action='store_true',
                        help="include to load all instances of data, omit to only load 3 instances")
    parser.add_argument("--prompting_type",
                        type=str,
                        default="base_TSBS",
                        help="type of prompting to use (base is 'Let's think step-by-step...')")
    parser.add_argument("--llama_prompt",
                        action='store_true',
                        help="include to use a template that follows the llama prompting guidelines ([SYS],[INST],etc.), omit otherwise")
    parser.add_argument("--output_formatting",
                        action='store_true',
                        help="include to use a template with output formatting instructions, omit otherwise")
    parser.add_argument("--chain_verbose",
                        action='store_true',
                        help="include to have verbose output from Langchain chain, omit otherwise")
    parser.add_argument("--save",
                        action='store_true',
                        help="include --save to save output to file, omit otherwise")

    args = parser.parse_args()
    main(args)