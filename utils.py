import pandas as pd

def preprocess(data):
    pass # Handle categorical stuff
    pass # Handle numerical stuff
    return data
    
def load_train():
    data = pd.read_csv('./data/Data-train.csv', index_col=0)
    data = preprocess(data)
    return data

def load_test():
    data = pd.read_csv('./data/Data-test.csv', index_col=0)
    data = preprocess(data)
    return data