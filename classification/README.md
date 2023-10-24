
# Binary Text Classification 


## Scripts

+ `classification/finetune.py` -- run to finetune BERT-based model (distilbert, bert-base, bert-large, roberta, deberta) on the task of classifying whether an explanations leads to a correct/incorrect prediction


+ `classification/lr_classifier.py` -- run to load the data (featurised and labels from evaluated responses), manipulate features (scaling, selection) according to the passed arguments, fit a Logistic Regression model and predict.


+ `classification/shap_aggregate_importances.py` -- run aggregation of shap values according to different strategies (specified as args)


+ `classification/shap_get_values.py` -- run shap to get shap values


# Directories

+ `classification/preds` -- predictions from models; used to calculate accuracy and agreement between different classifiers (see `analysis/classifiers_agreement.ipynb`)


+ `classification/models` -- the classifiers themselves


+ `classification/analysis` -- contains notebooks to analyse the results of the various experiments (with different features and models)


+ `classification/shap_values` -- contains the original shap values (output of shap) and the aggregated values (for different models and with different aggregation strategies)


+ `classification/split_datasets` -- dataset processed with HuggingFace's datasets library and used for finetuning and to ensure that all models use the same splits. Load by running `datasets.load_from_disk()`


+ `classification/stats` -- contains all stats related to features: exact features sets obtained by different combinations of selection methods, corresponding coefficients, metrics, ..