
'''
device('cuda') uses all gpus, set CUDA_VISIBLE_DEVICES before running script
CUDA_VISIBLE_DEVICES=0,1 python3 shap_get_values.py

DatasetDict({
    train: Dataset({
        features: ['text', 'label', 'pandas_idx'],
        num_rows: 8000
    })
    validation: Dataset({
        features: ['text', 'label', 'pandas_idx'],
        num_rows: 1000
    })
    test: Dataset({
        features: ['text', 'label', 'pandas_idx'],
        num_rows: 1000
    })
})
'''
    
import os
import shap
import torch
import argparse
import json
import pickle as pkl

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline

from datasets import load_from_disk


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "classification/models")


def main(args):

    # Load fine-tuned model
    path_to_model = os.path.join(MODELS_DIR, models_map[args.model])
    model = AutoModelForSequenceClassification.from_pretrained(path_to_model)
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model.to(device)

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        top_k=None, # get confidence scores for predictions
        # `return_all_scores` is now deprecated,  if want a similar funcionality use `top_k=None` instead of `return_all_scores=True` or `top_k=1` instead of `return_all_scores=False`.
    )
    pipe.device = device

    # Load data splits
    dataset = load_from_disk(os.path.join(SPLITS_DIR))
    if 'errors' in args.data_type:
        # Load errors map
        with open(os.path.join(PROJECT_DIR, "maps/errors_idx_uuid_map.json"), "r") as f:
            errors_map = json.load(f)
        errors_ids = [int(idx) for idx in errors_map.keys()]
        # Remove from dataset instances with pandas_idx in errors_ids
        dataset[args.split] = dataset[args.split].filter(
            lambda x: x['pandas_idx'] not in errors_ids
        )

    # Get SHAP values
    mask_tokens = [
        '[MASK]'] if args.mask_token == 'mask' else [
        ''] if args.mask_token == 'empty' else [
        '[MASK]', ''] if args.mask_token == 'both' else None

    for mask in mask_tokens:
        masker = shap.maskers.Text(tokenizer=r"\W+", collapse_mask_token=True)
        masker.mask_token = mask
        explainer = shap.Explainer(pipe, masker=masker, seed=1)

        shap_values = explainer(
            dataset[args.split]['text'] # list of strings
        ) # returns a shap explanation object

        filename = f"sv_{args.model}_{args.split}_{'empty' if mask=='' else 'mask'}.pkl"

        try:
            assert len(shap_values) == len(dataset[args.split]['text'])
        except AssertionError:
            print("SHAP values / dataset length mismatch for ", filename)

        with open(os.path.join(SHAP_DIR, filename), "wb") as f:
            pkl.dump(shap_values, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", 
                        type=str, 
                        default="test", 
                        choices=["train", "test"])
    parser.add_argument("--mask_token",
                        type=str,
                        default="empty",
                        choices=["mask", "empty", "both"])
    parser.add_argument("--model",
                        type=str,
                        required=True)
    parser.add_argument("--data_type",
                        type=str,
                        default="coqa")
                        
    args = parser.parse_args()

    data_name = args.data_type.split('/')[0] if '/' in args.data_type else args.data_type
    SPLITS_DIR = os.path.join(PROJECT_DIR, f"classification/split_datasets/{data_name}")
    SHAP_DIR = os.path.join(PROJECT_DIR, f"classification/shap_values/{args.data_type}") # output

    # Models trained on coqa (base)
    if data_name == 'coqa':
        models_map = {'distilbert': 'distilbert-base-uncased_13091207',
                    'bert': 'bert_14102004',
                    'bert-large': 'bert-large_14101938',
                    'roberta': 'roberta_14102014',
                    'deberta': 'deberta_14102242'}

    # Models trained on coqa_force_aug
    elif data_name == 'coqa_force_aug':
        models_map = {'bert': 'bert_06111925',
                    'bert_noerrors': 'bert_06111905'}
    
    main(args)