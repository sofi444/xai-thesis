
## stats/ naming conventions


`[generation run id]_[feature selection methods]_[type of features]_[number of instances]`


#### generation run id
+ 04091703: generation on 2000 instances from CommonsenseQA
+ 12091031: generation on 10000 instances from CommonsenseQA

#### feature selection methods
+ all: all features are kept == no feature selection
+ col: removes features that have collinearity above a define threshold (see utils.features)
+ rfe: Recursive Feature Elimination (see utils.features and https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
+ kbest: Select K Best features; based on either anova test (f_classification) or chi2 test (chi2) (see utils.features and https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)
+ ensemble: features are extracted with the methods listed but not sequentially. Each generates a feature set and then they are aggregate according to some criterion (all, most, any) (see utils.features)
+ interactions: interactions between fetures were calculated and added to the feature set

#### type of features
+ arg: only features from argument mining are used (from ArgQualityAdapters/)
+ nothing/trad: only traditional features are used (from tools in featureExtraction/)
+ all: both the feature types are used


**robust feature/interactions files** contain the features that pass VIF and p-value test (see `notebooks/vif_significance_on_existing_set.ipynb` and `notebooks/feature_interactions.ipynb`)