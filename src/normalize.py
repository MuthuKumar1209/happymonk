from sklearn.preprocessing import MinMaxScaler

def normalize_train_test(train_data,test_data):
    mm = MinMaxScaler()
    scaled_train_data = mm.fit_transform(train_data)
    scaled_test_data = mm.transform(test_data)
    return scaled_train_data,scaled_test_data