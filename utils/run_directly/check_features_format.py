
import os
import gzip
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(PROJECT_DIR, 'feature_extraction/features')

features_path = os.path.join(FEATURES_DIR, '02111129_all_features.csv.gz')

with gzip.open(features_path, 'rt') as f:
    df = pd.read_csv(f)

print(df.tail())

'''
   idx  nwords  Admiration/Awe_GALC  ...  respect_0  respect_1  respect_2
0    0     187                  0.0  ...   0.096214   0.807162   0.096623
1    1      89                  0.0  ...   0.009642   0.977206   0.013152
.
.
.
9998  9998     121                  0.0  ...   0.002623   0.961315   0.036062
9999  9999     125                  0.0  ...   0.007675   0.954257   0.038067
'''