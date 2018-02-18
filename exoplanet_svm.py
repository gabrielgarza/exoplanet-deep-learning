import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from pathlib import Path
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC
from scipy import ndimage, fft
from sklearn.preprocessing import normalize

from sklearn.externals import joblib


np.random.seed(1)

LOAD_MODEL = False # continue training previous weights or start fresh
RENDER_PLOT = False # render loss and accuracy plots

def X_Y_from_df(df):
    df = shuffle(df)
    df_X = df.drop(['LABEL'], axis=1)
    X = np.array(df_X)
    Y_raw = np.array(df['LABEL']).reshape((len(df['LABEL']),1))
    Y = Y_raw == 2
    return X, Y


def fourier_transform(X):
    return np.abs(fft(X, n=X.size))

if __name__ == "__main__":
    train_dataset_path = "./datasets/exoTrain.csv"
    dev_dataset_path = "./datasets/exoTest.csv"

    print("Loading datasets...")
    df_train = pd.read_csv(train_dataset_path, encoding = "ISO-8859-1")
    df_dev = pd.read_csv(dev_dataset_path, encoding = "ISO-8859-1")


    df_train_x = df_train.drop('LABEL', axis=1).apply(fourier_transform,axis=1)
    df_dev_x = df_dev.drop('LABEL', axis=1).apply(fourier_transform,axis=1)

    df_train_y = df_train.LABEL
    df_dev_y = df_dev.LABEL

    df_train_x = pd.DataFrame(normalize(df_train_x))
    df_dev_x = pd.DataFrame(normalize(df_dev_x))

    df_train_x = df_train_x.iloc[:,:(df_train_x.shape[1]//2)].values
    df_dev_x = df_dev_x.iloc[:,:(df_dev_x.shape[1]//2)].values

    df_train_x = ndimage.filters.gaussian_filter(df_train_x, sigma=10)
    df_dev_x = ndimage.filters.gaussian_filter(df_dev_x, sigma=10)

    df_train = pd.DataFrame(df_train_x).join(pd.DataFrame(df_train_y))
    df_dev = pd.DataFrame(df_dev_x).join(pd.DataFrame(df_dev_y))


    # Print number of positive exoplanet examples after augmenting data
    df_train_i = df_train['LABEL'] == 2
    print(np.sum(df_train_i))

    # Load X and Y
    X_train, Y_train = X_Y_from_df(df_train)
    X_dev, Y_dev = X_Y_from_df(df_dev)

    # Standardize X data
    print("Normalizing data...")
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

    # Build model or load model
    load_path="models_svm/svm_recall-1.0-1.0.pkl"
    my_file = Path(load_path)
    if LOAD_MODEL and my_file.is_file():
        model = joblib.load(load_path)
        print("------------")
        print("Loaded saved model")
        print("------------")
    else:
        print("------------")
        print("Building model")
        print("------------")
        model = LinearSVC()


    sm = SMOTE(ratio = 1.0)
    X_train_sm, Y_train_sm = sm.fit_sample(X_train, Y_train)

    # Train
    print("Training...")
    model.fit(X_train_sm, Y_train_sm)

    train_outputs = model.predict(X_train)
    dev_outputs = model.predict(X_dev)

    # Metrics
    train_outputs = model.predict(X_train)
    dev_outputs = model.predict(X_dev)
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
    classification_report_train = classification_report(Y_train, train_outputs)
    classification_report_dev = classification_report(Y_dev, dev_outputs)

    # Save model
    print("Saving...")
    save_path = "models_svm/svm_recall-{}-{}.pkl".format(recall_train, recall_dev)
    joblib.dump(model, save_path)

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
    print("classification_report_train")
    print(classification_report_train)
    print("classification_report_dev")
    print(classification_report_dev)
    print("------------")
    print("------------")
    print("Train Set Positive Predictions", np.count_nonzero(train_outputs))
    print("Dev Set Positive Predictions", np.count_nonzero(dev_outputs))
    #  Predicting 0's will give you error:
    print("------------")
    print("All 0's error train set", 37/5087)
    print("All 0's error dev set", 5/570)
    print("------------")
    print("------------")
