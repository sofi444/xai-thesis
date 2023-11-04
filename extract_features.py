"""
Feature extraction steps:

1. create csv for feature extraction (feature_extraction/prepare_csv.py)
2. create individual txt files for feature extraction (feature_extraction/prepare_feature_extraction_social_tools.py)
3. run SEANCE (feature_extraction/SEANCE_1_2_0_Py3/SEANCE_1_2_0.py)
4. run TAALED (feature_extraction/TAALED/TAALED_1_3_1_Py3/TAALED_1_4_1.py)
5. extract surface features (feature_extraction/featureExtraction/generate_surface_features.py)
6. extract syntax features (feature_extraction/featureExtraction/generate_syntactic_features.py)
7. merge all features (feature_extraction/featureExtraction/merge_features.py)
8. rename merged features file (add run id to filename)
"""


import subprocess as sp
import os
import shutil
import argparse


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_EXTRACTION_DIR = os.path.join(PROJECT_DIR, "feature_extraction")


def main(args):
    
    print(f"Extracting features from {args.in_file}")
    
    '''1. CREATE CSV FOR FEATURE EXTRACTION'''

    create_csv = sp.Popen([
        "python3", "feature_extraction/prepare_csv.py", 
        "--in_file", args.in_file
    ], 
    shell=False,
    ).wait()
    if create_csv != 0: # exit code (0 = success)
        raise Exception("Error creating csv file for feature extraction")
    
    csv_filename = args.in_file.split("/")[-1].split(".")[0] + ".csv"
    csv_filepath = os.path.join(FEATURE_EXTRACTION_DIR, "responses-fe", csv_filename)
    
    
    '''2. CREATE TXT FILES FOR FEATURE EXTRACTION'''

    create_txt_files = sp.Popen([
        "python3", "featureExtraction/prepare_feature_extraction_social_tools.py",
        "--input_path", csv_filepath,
        "--comment_column", "response",
        "--id_column", "idx"
    ], 
    shell=False,
    cwd=FEATURE_EXTRACTION_DIR
    ).wait()
    if create_txt_files != 0:
        raise Exception("Error creating txt files for feature extraction")
    
    
    '''3. RUN SEANCE'''

    print('''
    Running SEANCE...\n
    Interact with the UI\n
    \t1. Tick all boxes
    \t2. Set input dir to textfiles dir 
          (make sure it has the right files in it - remove old files)\n
    \t3. Set output filename to SEANCE_results.csv 
          (and make sure the path is correct)\n
    ''')

    run_seance = sp.Popen([
        "python3", "SEANCE_1_2_0.py"
    ],
    shell=False,
    cwd=os.path.join(FEATURE_EXTRACTION_DIR, "SEANCE_1_2_0_Py3")
    ).wait()
    if run_seance != 0:
        raise Exception("Error running SEANCE")
    
    seance_filepath = os.path.join(
        FEATURE_EXTRACTION_DIR, 
        "featureExtraction", 
        "output", 
        "SEANCE_results.csv"
    )
    
    
    '''4. RUN TAALED'''

    print('''
    Running TAALED...\n
    Interact with the UI\n
    \t1. Tick ‘all words’ + all features\n
    \t2. Set input dir to textfiles dir 
          (make sure it has the right files in it - remove old files)\n
    \t3.Set output filename to TAALED_results.csv 
          (and make sure the path is correct)\n
    ''')
    run_taaled = sp.Popen([
        "python3", "TAALED_1_4_1.py"
    ],
    shell=False,
    cwd=os.path.join(FEATURE_EXTRACTION_DIR, "TAALED", "TAALED_1_3_1_Py3")
    ).wait()
    if run_taaled != 0:
        raise Exception("Error running TAALED")
    
    taaled_filepath = os.path.join(
        FEATURE_EXTRACTION_DIR,
        "featureExtraction",
        "output",
        "TAALED_results.csv"
    )


    '''5. EXTRACT SURFACE FEATURES'''

    extract_surface_features = sp.Popen([
        "python3", "generate_surface_features.py",
        "--input_path", csv_filepath,
        "--output_path", "output/SURFACE_results.csv",
        "--comment_column", "response",
    ],
    shell=False,
    cwd=os.path.join(FEATURE_EXTRACTION_DIR, "featureExtraction")
    ).wait()
    if extract_surface_features != 0:
        raise Exception("Error extracting surface features")
    
    surface_features_filepath = os.path.join(
        FEATURE_EXTRACTION_DIR,
        "featureExtraction",
        "output",
        "SURFACE_results.csv"
    )


    '''6. EXTRACT SYNTAX FEATURES'''

    extract_syntax_features = sp.Popen([
        "python3", "generate_syntactic_features.py",
        "--input_path", csv_filepath,
        "--output_path", "output/SYNTAX_results.csv",
        "--comment_column", "response",
    ],
    shell=False,
    cwd=os.path.join(FEATURE_EXTRACTION_DIR, "featureExtraction")
    ).wait()
    if extract_syntax_features != 0:
        raise Exception("Error extracting syntax features")
    
    syntax_features_filepath = os.path.join(
        FEATURE_EXTRACTION_DIR,
        "featureExtraction",
        "output",
        "SYNTAX_results.csv"
    )


    '''7. MERGE ALL FEATURES'''

    merge_features = sp.Popen([
        "python3", "merge_features.py",
        "--input_path", csv_filepath,
        "--idcol", "idx",
        "--surface", surface_features_filepath,
        "--syntax", syntax_features_filepath,
        "--polarity", seance_filepath,
        "--taaled", taaled_filepath,
        "--output_path", "output/merged_features.csv",
        #"--filter" # include flag to filter by selected features list
    ],
    shell=False,
    cwd=os.path.join(FEATURE_EXTRACTION_DIR, "featureExtraction")
    ).wait()
    if merge_features != 0:
        raise Exception("Error merging features")

    
    '''8. MOVE & RENAME MERGED FEATURES FILE'''

    tmp_merged_features_filepath = os.path.join(
        FEATURE_EXTRACTION_DIR,
        "featureExtraction",
        "output",
        "merged_features.csv"
    )

    run_id = args.in_file.split("/")[-1].split("_")[0]
    new_features_filepath = os.path.join(
        FEATURE_EXTRACTION_DIR,
        "features",
        f"{run_id}_trad_features.csv"
    )
    shutil.move(tmp_merged_features_filepath, new_features_filepath)

    print(f"+++ All done!\n\tFinal features file: {new_features_filepath}")



    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_file",
                        type=str, 
                        help="file with text to extract features from")

    args = parser.parse_args()
    main(args)


