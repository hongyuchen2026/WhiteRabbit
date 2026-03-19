import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_processing import expand_labels, sliding_windows
from features import extract_features
from models import (train_random_forest, NeuralNetwork, FullyConnected, 
                    ReluActivation, SigmoidActivation, BinaryCrossEntropyLoss, Regularizer, LogisticRegression)


def main():




    # input data
    train_folder = "data/train/defog"
    train_folder1 = "data/train/defog/02ea782681.csv"
    window = 200
    step = 20

    all_features = []
    all_labels = []

    for fname in os.listdir(train_folder):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(train_folder, fname))

        df = expand_labels(df)
        segments, labels = sliding_windows(df, window, step)
        for seg, lab in zip(segments, labels):
            feat = extract_features(seg)
            all_features.append(feat)
            all_labels.append(lab)

    # df = pd.read_csv(train_folder1)
    # df = expand_labels(df)
    # segments, labels = sliding_windows(df, window, step)

    # for seg, lab in zip(segments, labels):
    #     feat = extract_features(seg)
    #     all_features.append(feat)
    #     all_labels.append(lab)


    X = np.array(all_features)
    y = np.array(all_labels).reshape(-1, 1)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    
    RandomForestClassifier_model = train_random_forest(X_train, y_train)
    y_pred1 = RandomForestClassifier_model.predict(X_test)
    print(classification_report(y_test, y_pred1))

    layers = [
    FullyConnected(16, 16, Regularizer()), ReluActivation(),
    FullyConnected(16, 16, Regularizer()), ReluActivation(),
    FullyConnected(16, 16, Regularizer()), ReluActivation(),
    FullyConnected(16, 1, Regularizer()), SigmoidActivation(),
    ]
    neural_network_model = NeuralNetwork(layers, BinaryCrossEntropyLoss(), learning_rate=0.1)
    neural_network_model.fit(X_train, y_train, max_iter=1000)
    y_pred2 = neural_network_model.predict(X_test)
    y_pred2 = (y_pred2 >= 0.5).astype(int).ravel()
    print(classification_report(y_test, y_pred2))

    logistic_regression_model = LogisticRegression(learning_rate=1e-1, max_iter=200)
    logistic_regression_model.fit(X_train, y_train)
    y_pred3 = logistic_regression_model.predict(X_test)
    print(classification_report(y_test, y_pred3))


if __name__ == "__main__":
    main()