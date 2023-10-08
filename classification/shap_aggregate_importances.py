
import os
import json
import pickle as pkl
from tqdm import tqdm
import numpy as np
import argparse
import re
import collections
from datasets import load_from_disk
import spacy
from spacy.tokenizer import Tokenizer



PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAP_DIR = os.path.join(PROJECT_DIR, "classification/shap_values/coqa")

dataset = load_from_disk(os.path.join(PROJECT_DIR, "classification/split_datasets/coqa"))['test']
map = json.load(open(os.path.join(PROJECT_DIR, "maps/idx_uuid_choices_map.json"), "r"))

spacy_pipe = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
spacy_pipe.tokenizer = Tokenizer(spacy_pipe.vocab, token_match=re.compile(r'\S+').match)



def add_contribution(shap_aggregation_dict, entity_shap_values, entity):
    ''' Aggregate SHAP values

    For binary classfication, SHAP values sum up to 0
    For each token, SHAP values have the same absolute value
    but with opposite sign for the two classes
    e.g. [-1.13922907e-04,  1.13922422e-04]
    Therefore, only store which class the token is important for
    (whether the value with positive sign is at index 0 or 1)
    and the magnitude of the SHAP value.

    Args:
        shap_aggregation_dict: {}
        entity_shap_values: [contribution to class 0, contribution to class 1]
        entity: token or ngram
    
    Returns:
        shap_aggregation_dict: {
            token: {
                'neg': [list of negative contributions], 
                'pos': [list of positive contributions]
            }
        }
    '''

    if entity not in shap_aggregation_dict:
        shap_aggregation_dict[entity] = {
            'neg': [],
            'pos': []
        }

    contribution_to_0 = entity_shap_values[0]
    contribution_to_1 = entity_shap_values[1]
    abs_shap_value = abs(contribution_to_0)

    if contribution_to_0 > contribution_to_1:
        # token contributes to prediction of class 0 (negative class)
        # => store negative contribution
        contribution = -abs_shap_value
        shap_aggregation_dict[entity]['neg'].append(contribution)
    elif contribution_to_0 < contribution_to_1:
        # token contributes to prediction of class 1 (positive class)
        # => store positive contribution
        contribution = abs_shap_value
        shap_aggregation_dict[entity]['pos'].append(contribution)

    return shap_aggregation_dict



def average_shap_values(per_class:bool, shap_aggregation_dict):
    ''' Average SHAP values

    Args:
        per_class: whether to average per class or overall

    Returns:
        avg_token_importance: {
            token: (avg neg contribution, avg pos contribution)
        }
    '''

    avg_token_importance = {}

    for token, values in shap_aggregation_dict.items():
        if per_class:
            # Average per class
            # Sum contributions to each class and divide by number of contributions to the respective class
            negative_contribution = round(
                sum(values['neg']) / len(values['neg']) if len(values['neg']) > 0 else 0,
                5)
            positive_contribution = round(
                sum(values['pos']) / len(values['pos']) if len(values['pos']) > 0 else 0,
                5)

            avg_token_importance[token] = (negative_contribution, positive_contribution)
        
        else:
            # Average over all contributions (regardless of class)
            # Sum up all contributions and divide by total number of contributions
            all_contributions = values['neg'] + values['pos']
            overall_average = round(
                sum(all_contributions) / len(all_contributions) if len(all_contributions) > 0 else 0,
                5)
            avg_token_importance[token] = overall_average
    
    return avg_token_importance



def create_ngrams(tokens, n):
    ''' Add padding and creat ngrams
    tokens: list of tokens (strings)
    n: ngram size

    num of PAD tokens = n-1
    
    returns: list of lists of ngrams (strings)
        [[w1, w2, w3], [w2, w3, w4], ...]]'''

    tokens = ["[BOS]"]*(n-1) + tokens + ["[EOS]"]*(n-1)
    ngrams = zip(*[tokens[i:] for i in range(n)])

    return ngrams



def clean_entity(entity, choices):
    ''' Normalize text entity
    (strip, lower, remove punctuation, lemmatise)
    
    entity: text (token | ngram | chunk)
    '''
    option_A = choices['A']
    other_options = f"{choices['B']}|{choices['C']}|{choices['D']}|{choices['E']}"
    
    entity = entity.replace("\n", " ").replace("\"", " ")
    entity = entity.replace("  ", " ").strip(" .,-\n\"")

    placeholder_map = collections.OrderedDict({
        "[OPT_L]": [
            # Sub option letters B-E when followed by punct or end of string
            re.compile(r"\b[B-E](\s*[.,:()-]+|\b|\Z)"),

            # Sub A only if followed by punct or by option text (lookahead)
            re.compile(r"\bA(\s*[.,:()-]+|\Z)"),
            re.compile(fr"\bA (?=\s*[.,:)-]+(?i:{option_A}))"),
        ],
        "[OPT_T]": [
            # Sub option text only if it follow option letter
            #re.compile(fr"(?<=\[OPT_L\])\s*(?i:({option_A+'|'+other_options}))"),

            # Always sub option text (no lookbehind)
            re.compile(fr"\b(?i:{option_A+'|'+other_options})\s*([.,:()]+|\Z)"),
        ],
        "[NMB]": [
            # Sub numbers
            re.compile(r"-?\d+[.,]?\d*[.]?\d*")
        ]
    })

    # Replace tokens with placeholders
    for placeholder, regex in placeholder_map.items():
        for r in regex:
            entity = re.sub(r, placeholder, entity)
    
    # Ensure space btw placeholders (diff tokens, diff values)
    entity = entity.replace("][", "] [")

    # Lemmatise
    # noun, plural | noun, proper plural | verb, 3rd person singular present
    lemma_tags = {"NNS", "NNPS", "VBZ"}
    special_tokens = list(placeholder_map.keys()) + ["[BOS]", "[EOS]"]

    if entity in special_tokens:
        return entity

    spacy_doc = spacy_pipe(entity)
    entity_out = []
    for token in spacy_doc:
        if token.text in special_tokens:
            entity_out.append(token.text.strip(" .,-\n\")(:?!"))
        elif token.tag_ in lemma_tags: # Sub token with its lemma
            entity_out.append(token.lemma_.strip(" .,-\n\")(:?!").lower())
        else:
            entity_out.append(token.text.strip(" .,-\n\")(:?!").lower())

    return " ".join(entity_out)



def main(args):

    # Load pre-calculated SHAP values
    with open(os.path.join(SHAP_DIR, args.sv_filename), "rb") as f:
        shap_values = pkl.load(f)
        # shape: (n_samples, n_features (None if n not fixed), n_classes)

    aggregated_token_importance = {}
    aggregated_ngram_importance = {}
    aggregated_chunk_importance = {}

    for i in tqdm(range(shap_values.shape[0])): # instances

        choices = map[str(dataset[i]['pandas_idx'])]['choices']
        tokens = list(shap_values[i].data)
        values = shap_values[i].values

        if args.aggregate_at_ngram_level:
            ngrams = create_ngrams(tokens, args.n) # adds padding and returns a list of ngrams (strings)

            # Get SHAP values for ngrams
            # ngram shap values = mean of shap values of its constituent tokens
            start_idx = 0
            for ngram in ngrams:
                ngram_shap_values = []
                padded_ngram = False

                for token in ngram:
                    if token in ["[BOS]", "[EOS]"]:
                        ngram_shap_values.append(np.array([0.0, 0.0]))
                        padded_ngram = True
                        continue
                    token_idx = tokens.index(token, start_idx)
                    ngram_shap_values.append(values[token_idx])
                
                # Only increment start_idx if no pad tokens (they don't have a corresponding idx)
                if not padded_ngram:
                    start_idx += 1
                
                assert len(ngram_shap_values) == len(ngram)

                ngram_shap_values = np.mean(ngram_shap_values, axis=0) # sum or mean?

                ngram = " ".join(ngram)
                ngram = clean_entity(entity=ngram, choices=choices).replace("  ", " ")

                aggregated_ngram_importance = add_contribution(
                    shap_aggregation_dict=aggregated_ngram_importance,
                    entity_shap_values=ngram_shap_values,
                    entity=ngram
                )


        if args.aggregate_at_token_level:
        
            for j in range(shap_values[i].shape[0]): # tokens
                    
                token = shap_values.data[i][j]
                token = clean_entity(entity=token, choices=choices)

                aggregated_token_importance = add_contribution(
                    shap_aggregation_dict=aggregated_token_importance,
                    entity_shap_values=shap_values.values[i][j],
                    entity=token
                )


        if args.aggregate_at_chunk_level:
            # text chunk = combination of tokens that have the same shap value
            # chunk's shap values = value of one of the tokens (their are all the same)

            current_chunk = ""
            current_values = None
            
            for j in range(shap_values[i].shape[0]):
                token = shap_values.data[i][j]
                values = shap_values.values[i][j]

                if current_values is None: # first token
                    current_values = values
                    current_chunk = token
                    continue
                
                if values[0] == current_values[0]: # same chunk
                    current_chunk += token # whitespace already in token (retain original spacing)
                
                else: # new chunk detected, add current chunk to dict
                    clean_chunk = clean_entity(entity=current_chunk, choices=choices)

                    aggregated_chunk_importance = add_contribution(
                        shap_aggregation_dict=aggregated_chunk_importance,
                        entity_shap_values=current_values,
                        entity=clean_chunk
                    )
                    # start new chunk
                    current_values = values
                    current_chunk = token


    if args.aggregate_at_ngram_level:
        class_avg_ngram_importance = average_shap_values(
            per_class=True,
            shap_aggregation_dict=aggregated_ngram_importance
        )
        overall_avg_ngram_importance = average_shap_values(
            per_class=False,
            shap_aggregation_dict=aggregated_ngram_importance
        )

    if args.aggregate_at_token_level:
        class_avg_token_importance = average_shap_values(
            per_class=True,
            shap_aggregation_dict=aggregated_token_importance
        )
        overall_avg_token_importance = average_shap_values(
            per_class=False,
            shap_aggregation_dict=aggregated_token_importance
        )

    if args.aggregate_at_chunk_level:
        class_avg_chunk_importance = average_shap_values(
            per_class=True,
            shap_aggregation_dict=aggregated_chunk_importance
        )
        overall_avg_chunk_importance = average_shap_values(
            per_class=False,
            shap_aggregation_dict=aggregated_chunk_importance
        )


    if args.save:
        if args.aggregate_at_token_level:
            try:
                with open(os.path.join(SHAP_DIR, "class-avg_token_importance_test.json"), "w") as f:
                    json.dump(class_avg_token_importance, f)
                with open(os.path.join(SHAP_DIR, "overall-avg_token_importance_test.json"), "w") as f:
                    json.dump(overall_avg_token_importance, f)
            except:
                print("Could not save TOKEN level shap values")
        
        if args.aggregate_at_ngram_level:
            try:
                with open(os.path.join(SHAP_DIR, f"class-avg_{args.n}gram_importance_test.json"), "w") as f:
                    json.dump(class_avg_ngram_importance, f)
                with open(os.path.join(SHAP_DIR, f"overall-avg_{args.n}gram_importance_test.json"), "w") as f:
                    json.dump(overall_avg_ngram_importance, f)
            except:
                print(f"Could not save NGRAM ({args.n}gram) level shap values")
        
        if args.aggregate_at_chunk_level:
            try:
                with open(os.path.join(SHAP_DIR, "class-avg_chunk_importance_test.json"), "w") as f:
                    json.dump(class_avg_chunk_importance, f)
                with open(os.path.join(SHAP_DIR, "overall-avg_chunk_importance_test.json"), "w") as f:
                    json.dump(overall_avg_chunk_importance, f)
            except:
                print("Could not save CHUNK level shap values")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sv_filename", type=str, required=True)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--aggregate_at_token_level", action="store_true")
    parser.add_argument("--aggregate_at_ngram_level", action="store_true")
    parser.add_argument("--aggregate_at_chunk_level", action="store_true")
    parser.add_argument("--n", type=int, default=3)

    args = parser.parse_args()
    main(args)
