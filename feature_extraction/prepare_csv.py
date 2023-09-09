import json
import os
import argparse
import pandas as pd



PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def jsonl_to_csv(args):
    '''
    Prepare text data for feature extraction (needs csv format)
    '''
    in_filepath = os.path.join(
        PROJECT_DIR, "responses", args.in_file
        ) if "/" not in args.in_file else os.path.join(PROJECT_DIR, args.in_file)
    
    with open((in_filepath), "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    column_names = ["idx", "response"]

    df = pd.DataFrame(columns=column_names)

    for idx, response in enumerate(data):
        clean_text = response["text"].replace("\n", "")
        df.loc[len(df)] = [response["idx"], clean_text]

    # save within feature_extraction dir
    OUT_DIR = os.path.join(PROJECT_DIR, "feature_extraction/responses-fe")

    if "/" in args.in_file:
        args.in_file = args.in_file.split("/")[-1]
    out_filepath = os.path.join(OUT_DIR, args.in_file.split(".")[0] + ".csv")

    df.to_csv(out_filepath, index=False)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_file",
                        type=str,
                        default="freetext_test.json",
                        help="filename of the json file to be converted to csv (responses)")
    
    args = parser.parse_args()
    jsonl_to_csv(args)
