
# Feature Extraction

#### ! When running any of the feature extraction tools make sure that the output path is `features_extraction/features/` (where all extracted features are stored)


`responses-fe` contains responses as .csv (needed for some feature extraction tools)


## ArgQualityAdapters

External repo; open-source; from argument quality research*

Contains pre-trained adapters that, from an input text, assign it a series of scores that represent different dimensions of argument quality.

*motivation and hypothesis: explanations share properties of arguments and therefore, existing tools to represent different aspects of arguments can be useful for representing explanations.

To extract all `arg` features, run `inference_parallel.py`


## featureExtraction

External repo; open-source

Uses SEANCE and TAALED, and additionally extracts surface and syntactic features.

It involves different steps and running several scripts. `xai-thesis/extract_features.py` makes it possible to run all the necessary step in one go (extracts `trad` features)


## SEANCE

Traditional open-source NLP tool that extracts linguistically motivated features from free text. The features are meant to measure linguistic aspects that are related to sentiment, cognition and social order. 

**Index-based** -- uses word (or word vector) dictionaries (i.e., indexes), which are lists of words representing different semantic categories.


## TAALED

Traditional open-source NLP tool that extracts features related to lexical diversity; **index-based**