
import openai
import transformers
import torch
import os
import subprocess as sp

from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load codellama

# models: openai_chat | llama2_chat | llama2_chat_local
def load_model(model:str="openai_chat", use_azure:bool=False, env_dict:dict=None):

    if model == "openai_chat":
        if use_azure:
            return AzureChatOpenAI(
                openai_api_base=env_dict["AZURE_OPENAI_ENDPOINT"],
                openai_api_version=env_dict["OPENAI_DEPLOYMENT_VERSION"],
                openai_api_key=env_dict["AZURE_OPENAI_KEY"],
                model=env_dict["OPENAI_MODEL_NAME"],
                deployment_name=env_dict["OPENAI_DEPLOYMENT_NAME"],
                temperature=0)
        else:
            return ChatOpenAI(model_name= "gpt-3.5-turbo-0613", # via azure we use 0301
                              temperature = 0)
    
    elif "llama2" in model:
        chat = True if "chat" in model else False
        local = True if "local" in model else False
        
        if local:
            call_llama2_chat_local() # tmp
        
        else:
            return LlamaCpp(
                model_path=env_dict[f"LLAMA2_13B_{'CHAT_' if chat else ''}PATH"],
                n_gpu_layers=1,
                n_batch=512,
                n_threads=8,
                f16_kv=True, # MUST be True, otherwise problems after a couple of calls (metal install)
                temperature=0.0,
                max_tokens=-1,
                repeat_penalty=1.1,
                verbose=True)
    
    elif model == "codellama":
        pass


def create_LLMchain(llm, prompt_template, verbose:bool=False):
    return LLMChain(
        llm=llm, 
        prompt=prompt_template, 
        verbose=verbose)


def call_llama2_chat_local():
    '''Calls the Llama2 chat model locally, using shell command '''

    LLAMA_DIR = "/Users/q616967/Workspace/llama.cpp"
    model_file = "llama-2-13b-chat.ggmlv3.q4_0.bin"

    # can't pass prompt as string, need to pass a filepath
    #prompts = load_templates("llama2_prompts.json")
    #sys_task = prompts["sys"]["CoQA_task"]
    #hum_instance = prompts["human"]["CoQA_instance"]
    #sys = "<<SYS>>\n{sys_task}\n<</SYS>>\n\n"
    #prompt = f"[INST]{sys}{hum_instance}[/INST]"

    PROMPT_DIR = os.path.join(PROJECT_DIR, "prompt_templates")
    prompt_path = os.path.join(PROMPT_DIR, "example_prompt_local_llama_CoQA.txt")

    run_llama = sp.Popen(
        f"cd {LLAMA_DIR} && LLAMA_METAL=1 make && ./ggml_metal.sh {model_file} {prompt_path}",
        shell=True,
        cwd=LLAMA_DIR,
        stdout=sp.PIPE,
        stderr=None
    )
    run_llama.wait()

    output = run_llama.communicate()[0]
    # cutoff_token must be bytes str (b"str")
    output = cutoff_text(output, cutoff_token=b"[/INST]", side_to_cutoff="R")
    output = remove_substring(str(output), "[/INST]").strip()

    print(output)
    print(run_llama.returncode)


# helper
def cutoff_text(text, cutoff_token, side_to_cutoff):
    """
    cutoff_token must be bytes (b'str') or int

    """
    idx = text.find(cutoff_token)

    if side_to_cutoff == "L":
        if idx != -1:
            return text[:idx] # return text up to cutoff phrase
        else:
            return text
    if side_to_cutoff == "R":
        if idx != 0:
            return text[idx:] # return from cutoff phrase on
        else:
            return text

# helper
def remove_substring(string, substring):
    return string.replace(substring, "")

# helper
def get_model_info(model_ob):
    return {'model_name': model_ob.model_name,
            'temperature': model_ob.temperature}
