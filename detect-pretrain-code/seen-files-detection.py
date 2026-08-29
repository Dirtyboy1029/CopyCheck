#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: DirtyBoy
@Date: 2026/8/6 21:31
"""
import os, json, argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def merge_bookname(data, data_type='book'):
    merged_data = defaultdict(list)
    for i, item in enumerate(data):
        if 'book' in data_type:
            merged_data[item['book']].append(i)
        elif 'arxiv' in data_type:
            merged_data[item['paper_id']].append(i)
        else:
            pass
    return dict(merged_data)


def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


if __name__ == '__main__':
    source_data = read_from_jsonl("datasets/bookmia_25_book.jsonl")
    bookname_snippets_indexs = merge_bookname(source_data)

    features = np.load("output/llama2-7b/bookmia_25_book.npy")
    print(features.shape)

    book_gt_labels = []
    X_features = []

    for book_name, snippets_indexs in bookname_snippets_indexs.items():
        book_gt_labels.append(source_data[snippets_indexs[0]]['label'])
        tmp_uc_metrics = np.max(features[snippets_indexs], axis=0)
        X_features.append(tmp_uc_metrics)

    X_features = np.array(X_features)

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X_features)
    outliers = gmm.predict(X_features)

    outliers = [(i + 1) % 2 for i in outliers]

    book_gt_labels = np.array(book_gt_labels)

    from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

    accuracy = accuracy_score(book_gt_labels, outliers)
    MSG = "The accuracy on the dataset is {:.2f}%"
    print(MSG.format(accuracy * 100))
    balanced_accuracy = balanced_accuracy_score(book_gt_labels, outliers)
    MSG = "The balanced accuracy on the dataset is {:.2f}%"
    print(MSG.format(balanced_accuracy * 100))
    tn, fp, fn, tp = confusion_matrix(book_gt_labels, outliers).ravel()
    print(f"True Negatives (tn): {tn} (Books correctly identified as 'unseen') / {tn + fp}")
    print(f"False Positives (fp): {fp} (Books incorrectly identified as 'unseen' but were 'seen') / {tn + fp}")
    print(f"False Negatives (fn): {fn} (Books incorrectly identified as 'seen' but were 'unseen') / {fn + tp}")
    print(f"True Positives (tp): {tp} (Books correctly identified as 'seen') / {fn + tp}")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(
        f"False Negative Rate (fnr): {fnr * 100:.2f}% (Books incorrectly identified as 'seen' but were 'unseen') / {fn + tp}")
    print(
        f"False Positive Rate (fpr): {fpr * 100:.2f}% (Books incorrectly identified as 'unseen' but were 'seen') / {fp + tn}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    print(f"F1 score: {f1 * 100:.2f}%")

    print(f"&{accuracy * 100:.2f} &{fnr * 100:.2f} &{fpr * 100:.2f} &{f1 * 100:.2f}")
