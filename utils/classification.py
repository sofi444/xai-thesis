
import pandas as pd

from sklearn.model_selection import train_test_split



def get_splits(data_df, splits_sizes=[0.8, 0.1, 0.1]):
    '''
    Split the dataset into train, validation and test sets.
    '''
    assert sum(splits_sizes) == 1, "Split sizes must sum to 1"

    train_size, val_size, test_size = splits_sizes
    train, tmp = train_test_split(data_df, test_size=val_size+test_size, random_state=42)
    val, test = train_test_split(tmp, test_size=test_size, random_state=42)

    assert len(train) + len(val) + len(test) == len(data_df), "Length mismatch (splits/original dataset)"
    print(f"Train: {len(train)}\nValidation: {len(val)}\nTest: {len(test)}")

    return train, val, test



def balance_dataset(data_df):
    '''
    Balance the dataset by downsampling the majority class (True)
    
    Returns a pandas dataframe with the same number of T and F instances
    '''

    true_instances_count = len(data_df[data_df['outcome'] == True])
    false_instances_count = len(data_df[data_df['outcome'] == False])
    #print(f'True instances: {true_instances_count}, False instances: {false_instances_count}')

    balanced_data_df = pd.concat(
        [data_df[data_df['outcome'] == True].sample(false_instances_count, random_state=1),
         data_df[data_df['outcome'] == False]]).sample(frac=1, random_state=1)

    print(f'Balanced dataset: {len(balanced_data_df)} instances.\n\tLength match? {len(balanced_data_df) == false_instances_count * 2}')

    return balanced_data_df