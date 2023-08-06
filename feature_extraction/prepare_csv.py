import json
import os
import argparse
import pandas as pd


def json_to_csv(args):

    with open(os.path.join(args.path, args.filename), "r") as f:
        json_data = json.load(f) # dict
    
    column_names = ["idx", "response"]

    # create pandas dataframe with the colum names in column_names
    # the values for the column 'idx' are the keys of the json dict
    # the values for the column 'response' are the values of the json dict
    df = pd.DataFrame(columns=column_names, data=json_data.items())
    
    # save in original output directory (?)
    csv_path = os.path.join(args.path, args.filename.split(".")[0] + ".csv")
    df.to_csv(csv_path, index=False)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filename",
        type=str,
        default="test_json.json",
        help="filename of the json file to be converted to csv (responses)"
    )

    parser.add_argument(
        "--path",
        type=str,
        default="/Users/q616967/Workspace/bmw/xai_sofia_casadei_master_thesis/responses",
        help="path to the json file. dafault is the responses folder in the project directory"
    )
    
    args = parser.parse_args()
    json_to_csv(args)
