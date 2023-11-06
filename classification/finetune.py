
import os
import numpy as np
import argparse
import datetime

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, set_seed
from transformers import DataCollatorWithPadding, default_data_collator
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import EarlyStoppingCallback
from datasets import load_from_disk

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(PROJECT_DIR, "classification/split_datasets")
MODELS_DIR = os.path.join(PROJECT_DIR, "classification/models")
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')
                                             

#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"


seed = 42
set_seed(seed)

label_map = {'False': 0, 'True': 1}
model_map = {
    'distilbert': 'distilbert-base-uncased',
    'deberta-small': 'microsoft/deberta-v3-small',
    'deberta': 'microsoft/deberta-v3-base',
    'roberta-small': 'roberta-small',
    'roberta': 'roberta-base',
    'bert': 'bert-base-uncased',
    'bert-large': 'bert-large-uncased'
}

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


class customTrainingArguments(TrainingArguments):
    def __init__(self,*args, **kwargs):
        super(customTrainingArguments, self).__init__(*args, **kwargs)

    @property
    #@torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        Name the device the number you use.
        """
        return torch.device("cuda:5")

    @property
    #@torch_required
    def n_gpu(self):
        """
        The number of GPUs used by this process.
        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        # _ = self._setup_devices
        # I set to one manullay
        self._n_gpu = 1
        return self._n_gpu


def tokenize_and_mask(raw_data):
    '''Tokenize
    Normal padding: set padding='max_length' and max_length=int (default is 512)
    Dynamic padding: set padding=False and (later in the Trainer) pass `data_collator=DataCollatorWithPadding(tokenizer)
    result will be a dict with keys 'input_ids', 'attention_mask'
    '''
    result = tokenizer(raw_data["text"],
                        max_length=512,
                        truncation=True,
                        #padding='max_length' # comment if using dynamic padding (set later)
                    )

    '''Add labels'''
    if label_map is not None:
        if "label" in raw_data:
            result['labels'] = [label_map[str(label)] for label in raw_data["label"]]
    
    return result


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(f'labels: {labels} -- distribution: {np.unique(labels, return_counts=True)}')
    print(f'preds: {preds} -- distribution: {np.unique(preds, return_counts=True)}')
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def get_report(preds, split, raw_dataset):
    y_trues = [0 if raw_dataset[split]['label'][i]==False else 1 for i in range(len(raw_dataset[split]))]
    y_preds = preds.predictions.argmax(-1)
    print(classification_report(y_trues, y_preds, labels=[0,1]))


def finetune(args):
    raw_dataset = load_from_disk(os.path.join(SPLITS_DIR, args.data_splits_dir))
    model_name = model_map[args.model]

    # get rid of config and pass num_labels=2 to model instead of config
    model_config = AutoConfig.from_pretrained(model_name,
                                              num_labels=len(label_map),
                                              hidden_dropout_prob=0.3,
                                              attention_probs_dropout_prob=0.3)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               config=model_config)
    model.to(device)

    '''Prepare inputs: tokenize and mask'''
    dataset = raw_dataset.map(tokenize_and_mask, batched=True)
    for split in ['train', 'validation', 'test']:
        dataset[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    data_collator_dynamic_padding = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    train_batch_size = 16
    eval_batch_size = 32
    steps_per_epoch = len(dataset['train']) // train_batch_size

    #training_args = TrainingArguments(
    training_args = customTrainingArguments( # use customArg to enforce using one GPU
        output_dir=LOGS_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=0.0001,
        warmup_steps=steps_per_epoch, # warm up for 1 epoch
        #weight_decay=0.05,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        save_strategy='steps',
        eval_steps=250,
        save_steps=500,
        logging_steps=250,
        logging_dir=LOGS_DIR,
        load_best_model_at_end=True, # needed for early stopping
        metric_for_best_model='accuracy', # needed for early stopping
        greater_is_better=True, # if metric is loss, set to False
        fp16=True,
        gradient_accumulation_steps=8, # virtual batch size (gr acc * batch size)
        seed=seed,
        full_determinism=True
    )

    total_steps = steps_per_epoch * training_args.num_train_epochs
    print(f"\nNumber of training steps: {total_steps}")
    print(f"Steps per epoch: {steps_per_epoch}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=training_args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6, # numerical stability
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=training_args.warmup_steps, 
        num_training_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args, # defined above or v1_training_args
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator_dynamic_padding, # default_data_collator or data_collator_dynamic_padding
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    print(trainer.args)
    trainer.train()

    ''' Eval '''
    for split in ['validation', 'test']:
        get_report(trainer.predict(dataset[split]), split, raw_dataset)

    ''' Save '''
    if args.save_model:
        model_id = datetime.datetime.now().strftime("%d%m%H%M")
        trainer.save_model(os.path.join(MODELS_DIR, f"{args.model}_{model_id}"))
        print(f"Saved model to {MODELS_DIR}/{args.model}_{model_id}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",
                        default="distilbert", 
                        type=str, 
                        choices=[
                            "distilbert", 
                            "deberta-small", 
                            "deberta", 
                            "roberta", 
                            "roberta-small", 
                            "bert",
                            "bert-large"
                        ])
    parser.add_argument("--save_model",
                        action="store_true")
    parser.add_argument("--data_splits_dir",
                        type=str,
                        required=True,
                        help="Name of directory with data splits (coqa | coqa_force)")
    
    args = parser.parse_args()
    finetune(args)
