import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from keras import metrics
from keras.callbacks import ModelCheckpoint

from pathlib import Path

from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import math
import time

# %matplotlib inline
np.random.seed(1)

# stdsc = StandardScaler()
# mms = MinMaxScaler()

LOAD_MODEL = False # continue training previous weights or start fresh
train_dataset_path = "./datasets/exoplanet_test_clean.csv"
dev_dataset_path = "./datasets/exoplanet_dev_clean.csv"


def train_test_data():
    # Load from csv

    # Separate X and Y

    # Return tensors

def standard_and_norm_data():


def build_network():
    model = Sequential()
    model.add(Dense(units=n_l1, input_dim=n_x))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=n_l2, input_dim=n_l1))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=n_y))
    model.add(Activation('sigmoid'))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # Load Dataset
    X, Y = load_data_w2v()

    X_train, X_test, Y_train, Y_test = train_test_data()

    (num_examples, n_x) = X_train.shape # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[1] # n_y : output size

    print("X_train.shape: ", X_train.shape)
    print("Y_train.shape: ", Y_train.shape)
    print("X_test.shape: ", X_test.shape)
    print("Y_test.shape: ", Y_test.shape)
    print("n_x: ", n_x)
    print("num_examples: ", num_examples)
    print("n_y: ", n_y)

    # Model config
    learning_rate = 0.001
    n_l1 = 20
    n_l2 = 20
    # Build model
    model = build_network()

    # Load weights
    # filepath="weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    # load_path="keras_ckpts/weights-best.hdf5"
    load_path="keras_ckpts/weights-acc-0.8629353848325511-0.858685296919657.hdf5"
    my_file = Path(load_path)
    if LOAD_MODEL and my_file.is_file():
        model.load_weights(load_path)
        print("loaded saved weights")

    # Train
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    history = model.fit(X_train, Y_train, epochs=200, batch_size=32)

    # Accuracy
    train_outputs = model.predict(X_train, batch_size=128)
    test_outputs = model.predict(X_test, batch_size=128)
    train_outputs = np.rint(train_outputs)
    test_outputs = np.rint(test_outputs)
    accuracy_train = accuracy_score(Y_train, train_outputs)
    accuracy_test = accuracy_score(Y_test, test_outputs)
    print("train error", 1.0 - accuracy_train)
    print("test error", 1.0 - accuracy_test)

    # Save weights
    save_path = "keras_ckpts/weights-acc-{}-{}.hdf5".format(accuracy_train, accuracy_test) # load_path
    model.save_weights(save_path)

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
