from pandas import get_dummies

def one_hot_encoding(data):
    print("performed one hot encoding on the dataset")
    return get_dummies(data)
