
import json
import os

from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate



PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



# helper
def load_templates(filename):
    filepath = os.path.join(PROJECT_DIR, "prompt_templates", filename)
    return json.load(open(filepath, "r"))


# helper
def get_llama_prompt(sys_prompt:str, user_message:str, i_first:bool=False):
    """
    Correct prompting for llama2 - template:

    <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_message }} [/INST]
    """
    llama_prompt = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{user_message} [/INST]"

    if i_first:
        llama_prompt = f"<s>[INST] {user_message}\n\n<<SYS>>\n{sys_prompt}\n<</SYS>> [/INST]"
    
    return llama_prompt


# helper
def compose_template(template_name:str, templates:dict, for_llama2:bool, i_first:bool):
    """
    Returns a template string composed of the sys prompt (the task) 
    and the instance (actual instance from the dataset).

    for_llama2: if True, follow the llama2 guidelines for formatting the prompt
    i_first: if True, the instance comes first, then the sys prompt
    """
    sys = templates["sys"][template_name]
    instance = templates["instance"]["dot_newl"]

    if for_llama2:
        return get_llama_prompt(sys, instance, i_first)
    
    return f"{instance}\n\n{sys}" if i_first else f"{sys}\n\n{instance}"
        


def create_template(prompting_type:str, dataset:str, output_formatting:bool, 
                    for_llama2:bool, i_first:bool=True):

    if dataset == "commonsenseQA":
        templates = load_templates("CoQA_templates.json")
        template_name = f"{prompting_type}_format" if output_formatting else prompting_type
        
        template = compose_template(template_name, templates, for_llama2, i_first) # str

        # chat - pass everything as human message
        #langchain_template = ChatPromptTemplate.from_template(template)
        # normal template
        langchain_template = PromptTemplate.from_template(template)
        
        return langchain_template
    
    else:
        raise NotImplementedError



def get_template_info(template_ob):
    try:
        content = template_ob.messages[0].prompt.template
    except:
        content = template_ob.template

    return {'content': content, 
            'input_variables': template_ob.input_variables}




if __name__ == "__main__":

    prompt = get_llama_prompt(sys_prompt="You are GoodBot. You answer always helpfully.",
                               user_message="What is the capital of Italy?",
                               i_first=False)

    template = create_template(prompting_type="base_TSBS",
                               dataset="commonsenseQA",
                               output_formatting=True,
                               for_llama2=True)
    
    #print(template)
    #print(type(template))
    #print(get_template_info(template))

    """
    prompt templates objects:

    <class 'langchain.prompts.prompt.PromptTemplate'>
    input_variables=['choice_A', 'choice_B', 'choice_C', 'choice_D', 'choice_E', 'format_instructions', 'question']
    output_parser=None partial_variables={} 
    template='[INST]<<SYS>>\nAnswer the question. Choose one of the five possible choices. Only one is correct. Think step by step.\n{format_instructions}\n\n\n<</SYS>>\n\nQuestion: {question}\n\nChoices:\nA={choice_A}; B={choice_B}; C={choice_C}; D={choice_D}; E={choice_E}[/INST]' 
    template_format='f-string'
    validate_template=True

    <class 'langchain.prompts.chat.ChatPromptTemplate'>
    input_variables=['choice_A', 'choice_B', 'choice_C', 'choice_D', 'choice_E', 'format_instructions', 'question'] 
    output_parser=None 
    partial_variables={} 
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['choice_A', 'choice_B', 'choice_C', 'choice_D', 'choice_E', 'format_instructions', 'question'], 
                output_parser=None, 
                partial_variables={}, 
                template='[INST]<<SYS>>\nAnswer the question. Choose one of the five possible choices. Only one is correct. Think step by step.\n{format_instructions}\n\n\n<</SYS>>\n\nQuestion: {question}\n\nChoices:\nA={choice_A}; B={choice_B}; C={choice_C}; D={choice_D}; E={choice_E}[/INST]',
                template_format='f-string', 
                validate_template=True), 
            additional_kwargs={})
            ]
    """