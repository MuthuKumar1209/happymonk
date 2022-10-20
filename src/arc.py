import tensorflow as tf

def model_build(INPUT,LABELS,AFS = ['relu','relu','softmax']):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(INPUT,activation=AFS[0],input_dim=INPUT),
    tf.keras.layers.Dense(8,activation=AFS[1]),
    tf.keras.layers.Dense(LABELS,activation=AFS[2])
    ])
    return model

def model_compile(model):
    return model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])

def model_train(model,train_data,e,bs)->None:
    if train_data is not None:
        return model.fit(*train_data,epochs = e,batch_size=bs)
    else:
        print("train data is none, please check the input parameters")
        exit()
