import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
print("TF Version: ", tf.__version__)
import keras

def process_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test

#flatten images
def flatten_image(X_train, X_val):
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_val_flat = X_val.reshape((X_val.shape[0], -1))
    return X_train_flat, X_val_flat

# One-hot encoding for target
def one_hot_target(y_train, y_val):
    y_train_enc = keras.utils.to_categorical(y_train, 10)
    y_val_enc = keras.utils.to_categorical(y_train, 10)
    return y_train_enc, y_val_enc

def keras_model():
    from keras.layers import Dense, Activation
    from keras.models import Sequential
    s = reset_tf_session() #clear a graph
    model = Sequential()
    model.add(Dense(256, input_shape=(784,))) #output array of shape 256
    model.add(Activation('sigmoid')) #sigmoid activation function
    model.add(Dense(256)) #after the first layer, you don't need to specify the size of the input anymore
    model.add(Activation('sigmoid'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    print(model.summary())
    return model

def compile_model(model, loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']):
    model = keras_model()
    model = model.compile(loss,
                          optimizer,
                          metrics)
    return model

def fit_keras_model(model, X_train_flat, y_train_enc, X_val_flat, y_val_enc, batch_size = 512, epochs = 40, verbose = 0):
    model = compile_model(model)
    model.fit(X_train_flat,
              y_train_enc,
              batch_size,
              epochs,
              validation_data = (X_val_flat, y_val_enc)#,
              #callbacks = []
              )
    return model
