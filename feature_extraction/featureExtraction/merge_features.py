import os
import pandas as pd
from tabulate import tabulate
from argparse import ArgumentParser


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_PREFIX = os.path.join(PROJECT_DIR, "feature_extraction/featureExtraction")


def merge_features(args):

    orig = pd.read_csv(os.path.join(PATH_PREFIX, args.input_path))

    '''Ignore polarity features. There is no script for this'''
    pol = pd.read_csv(os.path.join(PATH_PREFIX, args.polarity))
    # add column commentID to pol (it is column filename with .txt removed)
    pol[args.idcol] = pol["filename"].apply(lambda x: x[:-4])
    # convert idcol in taaled to int to avoid ValueError when merging
    pol[args.idcol] = pol[args.idcol].astype(int)
    # merge orig and pol on idcol, only keep columns from pol that are not in orig
    merged = pd.merge(orig, pol, on=args.idcol, how="left", suffixes=("", "_pol"))
    #merged = orig

    '''Ignore taales features. It does not work'''
    #taales = pd.read_csv(taales)
    #taales[idcol] = taales["Filename"].apply(lambda x: x[:-4])
    #merged = pd.merge(merged, taales, on=idcol, how="left", suffixes=("", "_taales"))

    '''TAALED features'''
    taaled = pd.read_csv(os.path.join(PATH_PREFIX, args.taaled))
    taaled[args.idcol] = taaled["filename"].apply(lambda x: x[:-4])
    # convert idcol in taaled to int to avoid ValueError when merging
    taaled[args.idcol] = taaled[args.idcol].astype(int)
    merged = pd.merge(merged, taaled, on=args.idcol, how="left", suffixes=("", "_taaled"))
    
    '''SURFACE and SYNTAX features'''
    surface = pd.read_csv(os.path.join(PATH_PREFIX, args.surface), sep="\t")
    # get column names that are not in merged already
    surface_cols = [col for col in surface.columns if col not in merged.columns]
    surface_cols.append(args.idcol)
    merged = pd.merge(merged, surface[surface_cols], on=args.idcol, how="left", suffixes=("", "_surface"))
    # merge on idcol but only add columns from surface that are not in merged
    
    syntax = pd.read_csv(os.path.join(PATH_PREFIX, args.syntax), sep="\t")
    syntax_cols = [col for col in syntax.columns if col not in merged.columns]
    syntax_cols.append(args.idcol)
    merged = pd.merge(merged, syntax[syntax_cols], on=args.idcol, how="left", suffixes=("", "_syntax"))
    
    #print(tabulate(merged.head(), headers="keys", tablefmt="psql"))

    if args.filter:
        merged = filter_dataframe(merged)

    #print("merged %d comments, %d features" % (len(merged), len(merged.columns)))
    #print(tabulate(merged.head(n=2), headers="keys", tablefmt="psql"))
    print(merged.head(n=2))
    
    print("writing merged data to %s" % os.path.join(PATH_PREFIX, args.output_path))
    merged.to_csv(os.path.join(PATH_PREFIX, args.output_path), index=False)



def filter_dataframe(df):
    """specify the corresponding columns for each feature set and filter out all columns that are not needed"""
    subset_diversity = open(
        os.path.join(PATH_PREFIX, "subset_diversity.txt")
        ).read().splitlines()
    subset_polarity = open(
        os.path.join(PATH_PREFIX, "subset_polarity.txt")
        ).read().splitlines()
    #subset_taales = open("subset_sophistication.txt").read().splitlines()
    # join the lists
    #subset = subset_diversity + subset_polarity + subset_taales
    subset = subset_diversity + subset_polarity

    # only keep columns that are in subset
    return df[subset]



if __name__ == '__main__':
    parser = ArgumentParser()

    '''All paths are relative to featureExtraction directory'''

    parser.add_argument("-i", "--input_path", dest="input_path", help="path to input file",
                        )
    parser.add_argument("-id", "--idcol", dest="idcol", help="name of id column",
                        default="idx")
    parser.add_argument("-su", "--surface", dest="surface", help="path to surface features",
                        default="output/SURFACE_results.csv")
    parser.add_argument("-sy", "--syntax", dest="syntax", help="path to syntax features",
                        default="output/SYNTAX_results.csv")
    parser.add_argument("-p", "--polarity", dest="polarity", help="path to polarity features",
                        default="output/SEANCE_results.csv")
    parser.add_argument("-t", "--taales", dest="taales", help="path to taales features",
                        default=None)
    parser.add_argument("-td", "--taaled", dest="taaled", help="path to taaled features",
                        default="output/TAALED_results.csv")
    parser.add_argument("-f", "--filter", dest="filter", action="store_true",
                        help="filter out columns that are not needed",
                        default=False)
    parser.add_argument("-o", "--output_path", dest="output_path", help="path to output file",
                        default="output/merged_features.csv")
    
    args = parser.parse_args()
    
    merge_features(args)

