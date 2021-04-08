from typing import List
import pandas as pd
import scipy as sp
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

random_state = 20

response = 'sales'

categorical = [
    'productID',
    'brandID',
    'weekday',
    'attribute1',
]

numerical = [
    'attribute2',
    'clickVolume',
    'avgOriginalUnitPrice',
    'avgFinalUnitPrice',
    'ma14SalesVolume',
    'meanAge',
    'gender',
    'meanEducation',
    'maritalStatus',
    'plus',
    'meanPurchasePower',
    'meanUserLevel',
    'meanCityLevel',
    # 'sales',
]

class Preprocessor:
    def __init__(self, encode_categorical, scale_numeric):
        self.encode_categorical = encode_categorical
        self.scale_numeric = scale_numeric
        self.encoder = OneHotEncoder(drop='first', sparse=False)
        self.sscaler = StandardScaler(with_mean=False, with_std=True)
    
    def fit(
        self, 
        data: pd.DataFrame, 
        categorical: List[str] = categorical, 
        numerical: List[str] = numerical,
    ):
        self.encoder.fit(data[categorical].apply(lambda x: x.astype('int')))
        self.sscaler.fit(data[numerical])
        return self
    
    def transform(
        self,
        data: pd.DataFrame, 
        categorical: List[str] = categorical, 
        numerical: List[str] = numerical,
    ):
        # To be safe
        data = data.copy()
        
        # Cast data types
        data[categorical] = data[categorical].apply(lambda x: x.astype('category'))
        data[numerical] = data[numerical].apply(lambda x: x.astype('float'))

        # Append dummies as new columns
        if self.encode_categorical:
            # data = pd.get_dummies(data, drop_first=False)
            encoded = self.encoder.transform(data[categorical])
            columns = self.encoder.get_feature_names(categorical)
            data = data.drop(columns=categorical)
            data[columns] = encoded
            # data = pd.concat([
            #     data.drop(columns=categorical),
            #     pd.DataFrame(encoded, columns=columns)
            # ], axis=1)
        
        # handle numerical data
        if self.scale_numeric:
            data[numerical] = self.sscaler.transform(data[numerical])

        return data
        
def load_datasets(train_file, test_file):
    """Utility function to load and preprocess standardized data"""
    # Split train-val-test split
    train_val = pd.read_csv(train_file, index_col=0)
    test = pd.read_csv(test_file, index_col=0)
    train, validation = train_test_split(train_val, test_size=0.2, random_state=random_state)

    #Further train-test split on train data for model tuning
    # train_train, validation_validation = train_test_split(train, test_size=0.2, random_state=random_state)

    
    # Fit preprocessor to train data only
    preprocessor = Preprocessor(encode_categorical=True, scale_numeric=True)
    preprocessor.fit(train)
    
    # Ensure all datasets undergo the same preprocessing steps
    train = preprocessor.transform(train)
    validation = preprocessor.transform(validation)
    test = preprocessor.transform(test)
    
    return train, validation, test

# Check that preprocessing steps are reproducible
foo, *_ = load_datasets(
    train_file = './data/Data-train.csv', 
    test_file = './data/Data-test.csv', 
)

bar, *_ = load_datasets(
    train_file = './data/Data-train.csv', 
    test_file = './data/Data-test.csv', 
)

assert all(foo == bar), 'Preprocessing not reproducible!'

# Module exports
train, validation, test = load_datasets(
    train_file = './data/Data-train.csv', 
    test_file = './data/Data-test.csv', 
)

y = response
X = train.columns.drop(response)

