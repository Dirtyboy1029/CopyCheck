# -*- coding: utf-8 -*- 
# @Time : 2025/1/16 10:57 
# 
# @File : myexp.py
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.utils import shuffle
import numpy as np


def save_model(clf, path):
    with open(path, 'wb') as file:
        pickle.dump(clf, file)


def load_model(path):
    with open(path, 'rb') as file:
        loaded_clf = pickle.load(file)
    return loaded_clf


def train_LogisticRegression(X_train: pd.DataFrame, y_train: list) -> LogisticRegression:
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf


def train_RandomForest(X_train: pd.DataFrame, y_train: list) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=3)
    clf.fit(X_train, y_train)
    return clf


def load_train_data(llm_type):
    x_train = pd.read_csv('X_feature/my_exp_trainset_' + llm_type + '.csv')
    y_train = []
    for n in range(40):
        path_to_labels = f'../data/final_chunks/open_llama_{llm_type}/my_exp_trainset_{n}_labels.pickle'
        with open(path_to_labels, 'rb') as f:
            labels = pickle.load(f)
        y_train += labels
    x_train.replace([np.inf, -np.inf], 0, inplace=True)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    return x_train, y_train


def load_test_data(testset, llm_type):
    x_test = pd.read_csv(f'X_feature/bonlyseen_{testset}_{llm_type}.csv')
    y_test = []
    for n in range(2):
        path_to_labels = f'../data/final_chunks/open_llama_{llm_type}/bonlyseen_{testset}_{n}_labels.pickle'
        with open(path_to_labels, 'rb') as f:
            labels = pickle.load(f)
        y_test += labels
    x_test.replace([np.inf, -np.inf], 0, inplace=True)
    return x_test, y_test


if __name__ == '__main__':
    testset = 40
    experiment_type = 'test'
    llm_type = '7b'
    if experiment_type == 'train':
        x_train, y_train = load_train_data(llm_type)
        model = train_RandomForest(x_train, y_train)
        save_model(model,  'rf_' + llm_type + '.pkl')
    elif experiment_type == 'test':
        x_test, y_test = load_test_data(testset, llm_type)
        model = load_model('rf_' + llm_type + '.pkl')
        y_pred = model.predict(x_test)
        from sklearn.metrics import confusion_matrix, accuracy_score

        balanced_accuracy = accuracy_score(y_test, y_pred)

        MSG = "The  accuracy on the dataset is {:.5f}%"
        print(MSG.format(balanced_accuracy * 100))
        tn, fp, fn, tp = confusion_matrix([1- item for item in y_test], [1- item for item in y_pred]).ravel()
        print(f"True Negatives (tn): {tn} (Books correctly identified as 'unseen') / {tn + fp}")
        print(f"False Positives (fp): {fp} (Books incorrectly identified as 'unseen' but were 'seen') / {tn + fp}")
        print(f"False Negatives (fn): {fn} (Books incorrectly identified as 'seen' but were 'unseen') / {fn + tp}")
        print(f"True Positives (tp): {tp} (Books correctly identified as 'seen') / {fn + tp}")
