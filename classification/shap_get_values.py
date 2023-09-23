
'''
device('cuda') uses all gpus, set CUDA_VISIBLE_DEVICES before running script
CUDA_VISIBLE_DEVICES=0,1 python3 shap_get_values.py
'''
    
import os
import shap
import torch
import pickle as pkl

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline

from datasets import load_from_disk


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


values_for_split = 'test' # 'train' | 'validation' | 'test'

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "classification/models")
SPLITS_DIR = os.path.join(PROJECT_DIR, "classification/split_datasets/coqa")
SHAP_DIR = os.path.join(PROJECT_DIR, "classification/shap_values")


''' Load fine-tuned model '''
path_to_model = os.path.join(MODELS_DIR, "distilbert-base-uncased_13091207")

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


''' Load data splits '''
dataset = load_from_disk(os.path.join(SPLITS_DIR))

'''
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


''' SHAP '''
explainer = shap.Explainer(pipe, seed=1)
shap_values = explainer(
    dataset[values_for_split]['text'] # list of strings
) # returns a shap explanation object

assert len(shap_values) == len(dataset[values_for_split]['text'])

with open(os.path.join(SHAP_DIR, f"{values_for_split}.pkl"), "wb") as f:
    pkl.dump(shap_values, f)