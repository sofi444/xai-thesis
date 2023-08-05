
import json
import datetime
import os

from langchain.output_parsers import StructuredOutputParser, ResponseSchema



def _get_schemas(schemas_src:str="default"):

    if schemas_src == "default":

        prediction_letter_schema = ResponseSchema(
            name="prediction_letter", 
            description="This is the answer to the question. It is one letter (either A, B, C, D, or E), which corresponds to the answer chosen by the model."
        )

        prediction_text_schema = ResponseSchema(
            name="prediction_text", 
            description="This is the answer to the question. It is the text of the answer chosen by the model."
        )

        explanation_schema = ResponseSchema(
            name="explanation", 
            description="This is the explanation for the prediction. It is a piece of text explains why the model chose the answer it did."
        )
    
        schemas = [prediction_letter_schema, prediction_text_schema, explanation_schema]
        return schemas
    
    else:
        raise NotImplementedError



def _create_output_parser(schemas, type:str="structured"):

    if type == "structured":
        return StructuredOutputParser.from_response_schemas(schemas)
    else:
        raise NotImplementedError



def get_format_instructions(schemas_src:str="default", parser_type:str="structured", only_json:bool=True):
    """
    Retrieves schemas, initializes output parser, and returns the parse and format instructions.

    only_json (bool): If True, only the json in the Markdown code snippet will be returned, 
    without the introducing text (the output should be a Markdown code ...). Defaults to False.
    """
    schemas = _get_schemas(schemas_src=schemas_src)
    output_parser = _create_output_parser(schemas=schemas,
                                         type=parser_type)
    format_instructions = output_parser.get_format_instructions(only_json=only_json)

    return output_parser, format_instructions



def write_responses(responses):
    """
    Write responses (dict) to a JSON file.
    Suitable for small experimental runs.
    """

    time_now = datetime.datetime.now()
    time_now_str = time_now.strftime("%d%m_%H%M")
    filename = f"responses_{time_now_str}.json"

    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(PROJECT_DIR, "outputs", filename)

    with open(filepath, "w") as f:
        json.dump(responses, f, indent=4)






if __name__ == "__main__":

    responses = {0: {'explanation': 'A revolving door is convenient for two direction travel, '
                    'but it also serves as a security measure at a bank. Banks '
                    'often use revolving doors to control the flow of people '
                    'entering and exiting the building, as well as to provide '
                    'an added layer of security by limiting access to the '
                    'interior of the bank.',
                    'prediction_letter': 'A',
                    'prediction_text': 'bank'},
                1: {'explanation': 'People aim to complete their job at work. This is the '
                                    'primary goal of most individuals in a professional '
                                    'setting. While learning from each other and talking to '
                                    'each other may also occur at work, the main objective is '
                                    'to complete the tasks and responsibilities assigned to '
                                    'them.',
                    'prediction_letter': 'A',
                    'prediction_text': 'complete job'},
                2: {'explanation': 'The correct answer is B: bookstore. Bookstores typically '
                                    'have a wide range of printed works, including magazines, '
                                    'available for purchase.',
                    'prediction_letter': 'B',
                    'prediction_text': 'bookstore'}}
    
    write_responses(responses)
    exit()

    format_instructions = get_format_instructions(
        schemas_src="default",
        parser_type="structured",
        only_json=True
    )

    print(format_instructions)



