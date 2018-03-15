import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras import metrics
from keras.callbacks import ModelCheckpoint

from imblearn.over_sampling import SMOTE

from pathlib import Path

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import math
import time

from sklearn.metrics import classification_report

from  scipy import ndimage, fft
from sklearn.preprocessing import normalize

from preprocess_data import LightFluxProcessor

np.random.seed(1)

LOAD_MODEL = True # continue training previous weights or start fresh
RENDER_PLOT = False # render loss and accuracy plots

def X_Y_from_df(df):
    df = shuffle(df)
    df_X = df.drop(['LABEL'], axis=1)
    X = np.array(df_X)
    Y_raw = np.array(df['LABEL']).reshape((len(df['LABEL']),1))
    Y = Y_raw == 2
    return X, Y

def build_network():
    # Model config
    learning_rate = 0.001

    layers = [
        { "units": 1, "input_dim": n_x, "activation": 'relu', "dropout": 0 },
        { "units": n_y, "input_dim": 1, "activation": 'sigmoid', "dropout": 0 },
    ]

    # Build model
    model = Sequential()
    for layer in layers:
        model.add(Dense(units=layer["units"], input_dim=layer["input_dim"]))
        model.add(Activation(layer["activation"]))
        if layer["dropout"] > 0:
            model.add(Dropout(layer["dropout"]))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    return model

def fourier_transform(X):
    spectrum = fft(X, n=X.size)
    return np.abs(spectrum)

if __name__ == "__main__":
    train_dataset_path = "./datasets/exoTrain.csv"
    dev_dataset_path = "./datasets/exoTest.csv"

    print("Loading datasets...")
    df_train = pd.read_csv(train_dataset_path, encoding = "ISO-8859-1")
    df_dev = pd.read_csv(dev_dataset_path, encoding = "ISO-8859-1")

    # Process dataset
    LFP = LightFluxProcessor(
        fourier=True,
        normalize=True,
        gaussian=True)
    df_train, df_dev = LFP.process(df_train, df_dev)

    # Load X and Y
    X_train, Y_train = X_Y_from_df(df_train)
    X_dev, Y_dev = X_Y_from_df(df_dev)

    # Standardize X data
    print("Standardizing data...")
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_dev = std_scaler.transform(X_dev)

    # Print data set stats
    (num_examples, n_x) = X_train.shape # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[1] # n_y : output size
    print("X_train.shape: ", X_train.shape)
    print("Y_train.shape: ", Y_train.shape)
    print("X_dev.shape: ", X_dev.shape)
    print("Y_dev.shape: ", Y_dev.shape)
    print("n_x: ", n_x)
    print("num_examples: ", num_examples)
    print("n_y: ", n_y)

    # Build model
    model = build_network()

    # Load weights
    load_path=""
    my_file = Path(load_path)
    if LOAD_MODEL and my_file.is_file():
        model.load_weights(load_path)
        print("------------")
        print("Loaded saved weights")
        print("------------")


    sm = SMOTE(ratio = 1.0)
    X_train_sm, Y_train_sm = sm.fit_sample(X_train, Y_train)

    # Train
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    print("Training...")
    history = model.fit(X_train_sm, Y_train_sm, epochs=50, batch_size=32)

    # Metrics
    train_outputs = model.predict(X_train, batch_size=32)
    dev_outputs = model.predict(X_dev, batch_size=32)
    train_outputs = np.rint(train_outputs)
    dev_outputs = np.rint(dev_outputs)
    accuracy_train = accuracy_score(Y_train, train_outputs)
    accuracy_dev = accuracy_score(Y_dev, dev_outputs)
    precision_train = precision_score(Y_train, train_outputs)
    precision_dev = precision_score(Y_dev, dev_outputs)
    recall_train = recall_score(Y_train, train_outputs)
    recall_dev = recall_score(Y_dev, dev_outputs)
    confusion_matrix_train = confusion_matrix(Y_train, train_outputs)
    confusion_matrix_dev = confusion_matrix(Y_dev, dev_outputs)

    # Save model
    print("Saving model...")
    save_weights_path = "checkpoints_v2/weights-recall-{}-{}.hdf5".format(recall_train, recall_dev) # load_path
    model.save_weights(save_weights_path)
    save_path = "models_v2/model-recall-{}-{}.hdf5".format(recall_train, recall_dev) # load_path
    # model.save(save_path)

    print("train set error", 1.0 - accuracy_train)
    print("dev set error", 1.0 - accuracy_dev)
    print("------------")
    print("precision_train", precision_train)
    print("precision_dev", precision_dev)
    print("------------")
    print("recall_train", recall_train)
    print("recall_dev", recall_dev)
    print("------------")
    print("confusion_matrix_train")
    print(confusion_matrix_train)
    print("confusion_matrix_dev")
    print(confusion_matrix_dev)
    print("------------")
    print("Train Set Positive Predictions", np.count_nonzero(train_outputs))
    print("Dev Set Positive Predictions", np.count_nonzero(dev_outputs))
    #  Predicting 0's will give you error:
    print("------------")
    print("All 0's error train set", 37/5087)
    print("All 0's error dev set", 5/570)

    print("------------")
    print("------------")

    if RENDER_PLOT:
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
