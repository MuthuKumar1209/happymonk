import numpy as np
from src.loader import *
from src.encode import *
from src.arc import *
from src.normalize import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from itertools import combinations_with_replacement
import pandas as pd

# Activation function list
LIST_ACTIVATION = ['relu','sigmoid','elu','selu','tanh','exponential','softmax']
tmp = list(combinations_with_replacement(LIST_ACTIVATION,3))


df = load_data(data_path="datasets/iris.csv")
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

trainx,testx,trainy,testy = train_test_split(x,y,test_size=0.3)
encoded_trainy = one_hot_encoding(trainy)
encoded_testy = one_hot_encoding(testy)
scaled_trainx,scaled_testx = normalize_train_test(train_data=trainx,test_data=testx)

results = {"activation_list":[],"train_accuracy":[],"test_accuracy":[]}

for ACTIVATION_FUNCTION_LIST in tmp:
    if ACTIVATION_FUNCTION_LIST[-1]=="softmax" and ACTIVATION_FUNCTION_LIST[0] != "softmax" and ACTIVATION_FUNCTION_LIST[1]!="softmax":
        model = model_build(trainx.shape[1],encoded_trainy.shape[1],AFS = ACTIVATION_FUNCTION_LIST)
        model_compile(model)
        model_train(model=model,train_data=[scaled_trainx,encoded_trainy],e=10,bs=2)
        results["activation_list"].append(ACTIVATION_FUNCTION_LIST)
        pred = model.predict(scaled_trainx)
        results["train_accuracy"].append(accuracy_score(np.argmax(encoded_trainy.values,axis=1),np.argmax(pred,axis=1)))
        pred = model.predict(scaled_testx)
        results["test_accuracy"].append(accuracy_score(np.argmax(encoded_testy.values,axis=1),np.argmax(pred,axis=1)))


results = pd.DataFrame(results)
results["Diff"] = np.abs(results["train_accuracy"] - results["test_accuracy"])
results.sort_values(by="Diff",ascending=True).to_csv("results/results.csv",index=False)