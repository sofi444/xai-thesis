
import os
import json
import pickle as pkl
from tqdm import tqdm
import numpy as np
import argparse


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAP_DIR = os.path.join(PROJECT_DIR, "classification/shap_values/coqa")



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
    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngrams = [" ".join(ngram) for ngram in ngrams]

    return ngrams



def main(args):

    # Load pre-calculated SHAP values
    with open(os.path.join(SHAP_DIR, "test.pkl"), "rb") as f:
        shap_values = pkl.load(f) 
        # shape: (n_samples, n_features (None if n not fixed), n_classes)

    aggregated_token_importance = {}
    aggregated_ngram_importance = {}

    for i in tqdm(range(shap_values.shape[0])): # instances
        
        if args.aggregate_at_ngram_level:
            tokens = shap_values[i].data
            tokens = [token.strip().lower() for token in tokens]
            values = shap_values[i].values
            ngrams = create_ngrams(tokens, args.n)
            current_idx = 0

            for ngram in ngrams:
                ngram_tokens = ngram.split()
                ngram_idxs = []
                for token in ngram_tokens:
                    token_idx = tokens.index(token, current_idx)
                    ngram_idxs.append(token_idx)
                    current_idx = token_idx

                if ngram.startswith(" "):
                    ngram_tokens = ["<BOS>"] + ngram_tokens
                    ngram_idxs = [0] + ngram_idxs
                    ngram = "<BOS>" + ngram
                if ngram.endswith(" "):
                    ngram_tokens = ngram_tokens + ["<EOS>"]
                    ngram_idxs = ngram_idxs + [len(tokens)-1]
                    ngram = ngram + "<EOS>"
                
                # get shap values for ngram
                ngram_shap_values = values[ngram_idxs[0]:ngram_idxs[-1]+1]
                ngram_shap_values = np.mean(ngram_shap_values, axis=0) # sum or mean?

                aggregated_ngram_importance = add_contribution(
                    shap_aggregation_dict=aggregated_ngram_importance,
                    entity_shap_values=ngram_shap_values,
                    entity=ngram
                )


        if args.aggregate_at_token_level:
        
            for j in range(shap_values[i].shape[0]): # tokens
                    
                token = shap_values.data[i][j]
                token = token.strip().lower()

                aggregated_token_importance = add_contribution(
                    shap_aggregation_dict=aggregated_token_importance,
                    entity_shap_values=shap_values.values[i][j],
                    entity=token
                )


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



    if args.save:
        try:
            with open(os.path.join(SHAP_DIR, "class-avg_token_importance_test.json"), "w") as f:
                json.dump(class_avg_token_importance, f)
            with open(os.path.join(SHAP_DIR, "overall-avg_token_importance_test.json"), "w") as f:
                json.dump(overall_avg_token_importance, f)
        except:
            print("Could not save TOKEN level shap values")
        
        try:
            with open(os.path.join(SHAP_DIR, f"class-avg_{args.n}gram_importance_test.json"), "w") as f:
                json.dump(class_avg_ngram_importance, f)
            with open(os.path.join(SHAP_DIR, f"overall-avg_{args.n}gram_importance_test.json"), "w") as f:
                json.dump(overall_avg_ngram_importance, f)
        except:
            print(f"Could not save NGRAM ({args.n}gram) level shap values")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--aggregate_at_token_level", action="store_true")
    parser.add_argument("--aggregate_at_ngram_level", action="store_true")
    parser.add_argument("--n", type=int, default=2)

    args = parser.parse_args()
    main(args)
