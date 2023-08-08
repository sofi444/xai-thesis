import json
import os
import argparse
import pandas as pd



def json_to_csv(args):
    if args.dir_path is None:
        # path of project directory
        PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(PROJECT_DIR, "responses", args.filename)
    else:
        filepath = os.path.join(args.dir_path, args.filename)

    with open((filepath), "r") as f:
        json_data = json.load(f) # dict

    column_names = ["idx", "response"]

    if args.freetext:
        # create pandas dataframe with the colum names in column_names
        # the values for the column 'idx' are the keys of the json dict
        # the values for the column 'response' are the values of the json dict
        df = pd.DataFrame(columns=column_names, data=json_data.items())
    else:
        # create pandas df from formatted resonses
        df = pd.DataFrame(columns=column_names)
        for idx, response in json_data.items():
            df.loc[len(df)] = [idx, response["explanation"]]

    # save within feature_extraction dir
    OUT_DIR = os.path.join(PROJECT_DIR, "feature_extraction/responses-fe")
    out_path = os.path.join(OUT_DIR, args.filename.split(".")[0] + ".csv")
    df.to_csv(out_path, index=False)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filename",
        type=str,
        default="freetext_test.json",
        help="filename of the json file to be converted to csv (responses)"
    )

    parser.add_argument(
        "--dir_path",
        type=str,
        default=None,
        help="path to the json file. dafault is the responses folder in the project directory"
    )

    parser.add_argument(
        "--freetext",
        action="store_true",
        help="include --freetext if the response are freetext, omit if they are formatted"
    )
    
    args = parser.parse_args()
    json_to_csv(args)
