
import pandas as pd
import os

filepath = os.path.join(os.path.dirname(__file__), 'output_12091031_all.csv')
df = pd.read_csv(filepath, sep='\t')

"""
10000 rows x 34 columns

df.columns:
    ['idx', 'response', 'reasonableness', 'effectiveness', 'overall',
    'impact_0', 'impact_1', 'impact_2', 'quality', 'clarity',
    'justification_0', 'justification_1', 'justification_2',
    'justification_3', 'interactivity_0', 'interactivity_1',
    'interactivity_2', 'interactivity_3', 'cgood_0', 'cgood_1', 'cgood_2',
    'story', 'reference', 'posEmotion', 'negEmotion', 'empathy',
    'argumentative', 'narration', 'proposal', 'QforJustification',
    'cogency', 'respect_0', 'respect_1', 'respect_2']
"""

# check if there are any NaN values
print(df.isnull().values.any()) # False
# check unique values per column
print(df.nunique())

print(df.describe())