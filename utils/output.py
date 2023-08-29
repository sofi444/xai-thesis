
import json
import datetime as dt
import os

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser, PydanticOutputParser

from pydantic import BaseModel, Field, validator



PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def load_schemas(schemas_version:str):
    schemas_filepath = os.path.join(PROJECT_DIR,
                                    "schemas/output-parsing_schemas.json")
    with open(schemas_filepath, "r") as f:
        schemas = json.load(f)
        schemas = schemas[schemas_version]
    
    return schemas



def get_langchain_schemas(schemas):
    return [ResponseSchema(name=k, description=v) for k,v in schemas.items()]



class PydanticResponse(BaseModel): # not in use

    answer_letter: str = Field(description="the letter corresponding to the final answer.")
    answer_text: str = Field(description="the text corresponding to the final answer.")

    # You can add custom validation logic easily with Pydantic.
    @validator("answer_letter")
    def is_valid_choice(cls, field):
        if field not in ["A", "B", "C", "D", "E"]:
            raise ValueError("Not a valid choice")
        return field
    


def build_parser(schemas_version:str="v2", parser_type:str="structured", only_json:bool=False):
    """
    Retrieves schemas, initializes output parser, and returns the parse and format instructions.

    only_json (bool): If True, only the json in the Markdown code snippet will be returned, 
    without the introducing text (the output should be a Markdown code ...). Defaults to False.
    """

    schemas = get_langchain_schemas(load_schemas(schemas_version))

    if parser_type == "structured":
        output_parser = StructuredOutputParser.from_response_schemas(schemas)
        format_instructions = output_parser.get_format_instructions(only_json=only_json)

    elif parser_type == "pydantic": # not in use
        raise NotImplementedError("Use structured parser")
        output_parser = PydanticOutputParser(pydantic_object=PydanticResponse)
        format_instructions = output_parser.get_format_instructions()

    return output_parser, format_instructions


def get_run_id():
    return dt.datetime.now().strftime("%d%m%H%M")


def write_all_responses(responses, run_id, idx_uuid_map=None):
    """
    Write one dict containg all responses from a run to json file
    Format: {idx: freetext response, ...}
    save mapping of idx (from run) to uuid (from dataset)
    """
    filepath = os.path.join(PROJECT_DIR, 
                            f"responses/responses_{run_id}.json")

    with open(filepath, "w") as f:
       json.dump(responses, f, indent=4)

    if idx_uuid_map:
        filepath_uuids = os.path.join(PROJECT_DIR, "processed_uuids", f"uuids_{run_id}.json")
        with open(filepath_uuids, "w+") as f:
            json.dump(idx_uuid_map, f, indent=4)


def write_batch_responses(responses, run_id):
    """
    Write responses in one batch to a txt file
    Format: [{uuid: uuid, id: idx, response: freetext response}, ...]
    """
    filepath = os.path.join(PROJECT_DIR, 
                            f"responses/responses_batch_{run_id}.txt")

    with open(filepath, "a+") as f:
        for response in responses:
            f.write(response)
            f.write("\n")



if __name__ == "__main__":

    output_parser, format_instructions = build_parser(parser_type="structured")

    print(format_instructions)



