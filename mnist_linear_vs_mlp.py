import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
%matplotlib inline
import tensorflow as tf
from collections import defaultdict
import numpy as np
from keras.models import save_model
import tensorflow as tf
import keras
from keras import backend as K

def reset_tf_session():
    curr_session = tf.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    K.clear_session()
    # create new session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    K.set_session(s)
    return s

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

def flatten_image(X_train, X_val):
    #flatten images
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_val_flat = X_val.reshape((X_val.shape[0], -1))
    return X_train_flat, X_val_flat

def one_hot_target(y_train, y_val):
    # One-hot encoding for target
    y_train_enc = keras.utils.to_categorical(y_train, 10)
    y_val_enc = keras.utils.to_categorical(y_train, 10)
    return y_train_enc, y_val_enc

def_model_params(shape1 = 784, shape2 = 10, dtype=tf.float32):
    # Model parameters: W and b
    W = tf.get_variable("W", shape=(shape1, shape2), dtype = dtype)
    b = tf.get_variable("b", shape=(shape2), dtype=dtype)
    return W, b

def input_values(shape_a=None, shape1b=784, shape2b=10, dtype="float32"):
    # Placeholders for the input data
    input_X = tf.placeholder(dtype, shape=(shape_a, shape1b))
    input_y = tf.placeholder(dtype, shape=(shape_a,shape2b))
    return input_X, input_y

def training_attributes_tf(labels=input_y, learning_rate=0.003):
    logits = tf.matmul(input_X, W) + b
    probas = tf.nn.softmax(logits)
    classes = tf.argmax(probas, axis=1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits))
    step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return loss, step

def training_attributes_mlp(labels=input_y, learning_rate=0.003, activation=tf.nn.sigmoid):
    logits = tf.layers.dense(input_X, 786, activation = activation)
    logits2 = tf.layers.dense(logits,256,activation = activation)
    logits3 = tf.layers.dense(logits2,256,activation = activation)
    logits4 = tf.layers.dense(logits,10,activation = activation)
    probas = tf.nn.softmax(logits4)
    classes = tf.argmax(probas,axis = 1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = input_y,logits = logits4))
    step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return loss, step


def run_training(batch_size = 512, epochs = 40, step, loss):
    s.run(tf.global_variables_initializer())
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    for epoch in range(EPOCHS):
        batch_losses = []
        for batch_start in range(0, X_train_flat.shape[0], BATCH_SIZE):  # data is already shuffled
            _, batch_loss = s.run([step, loss], {input_X: X_train_flat[batch_start:batch_start+BATCH_SIZE],
                                                 input_y: y_train_enc[batch_start:batch_start+BATCH_SIZE]})
            # collect batch losses, this is almost free as we need a forward pass for backprop anyway
            batch_losses.append(batch_loss)
        train_loss = np.mean(batch_losses)
        val_loss = s.run(loss, {input_X: X_val_flat, input_y: y_val_enc})
        train_accuracy = accuracy_score(y_train, s.run(classes, {input_X: X_train_flat}))
        valid_accuracy = accuracy_score(y_val, s.run(classes, {input_X: X_val_flat}))
        return train_loss, val_loss, train_accuracy, valid_accuracy
