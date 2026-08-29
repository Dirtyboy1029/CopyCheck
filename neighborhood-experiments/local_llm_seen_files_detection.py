#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: DirtyBoy
@Date: 2026/8/6 21:50
"""
import os, json, argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


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


def find_best_threshold(
        features,
        labels,
        metric="accuracy",
        greater_is_positive=True
):
    features = np.asarray(features, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)

    if len(features) != len(labels):
        raise ValueError("features 和 labels 的长度必须相同。")

    if len(features) == 0:
        raise ValueError("输入数组不能为空。")

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels 中只能包含 0 和 1。")

    valid_metrics = {
        "accuracy",
        "f1",
        "precision",
        "recall",
        "youden"
    }

    unique_values = np.sort(np.unique(features))

    if len(unique_values) == 1:
        thresholds = unique_values
    else:
        middle_points = (
                                unique_values[:-1] + unique_values[1:]
                        ) / 2

        # 加入两端阈值，保证“全部预测为0/1”的情况也会被检查
        lower = np.nextafter(unique_values[0], -np.inf)
        upper = np.nextafter(unique_values[-1], np.inf)

        thresholds = np.concatenate([
            [lower],
            middle_points,
            [upper]
        ])

    best_threshold = None
    best_score = -np.inf
    best_predictions = None

    for threshold in thresholds:
        if greater_is_positive:
            predictions = (features >= threshold).astype(int)
        else:
            predictions = (features <= threshold).astype(int)

        if metric == "accuracy":
            score = accuracy_score(labels, predictions)

        elif metric == "f1":
            score = f1_score(
                labels,
                predictions,
                zero_division=0
            )

        elif metric == "precision":
            score = precision_score(
                labels,
                predictions,
                zero_division=0
            )

        elif metric == "recall":
            score = recall_score(
                labels,
                predictions,
                zero_division=0
            )

        else:
            tn, fp, fn, tp = confusion_matrix(
                labels,
                predictions,
                labels=[0, 1]
            ).ravel()

            tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
            fpr = fp / (fp + tn) if fp + tn > 0 else 0.0

            score = tpr - fpr

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_predictions = predictions.copy()

    return best_threshold, best_predictions


if __name__ == '__main__':
    llm = "llama2-7b"
    feature = "p"
    source_data = read_from_jsonl("../detect-pretrain-code/datasets/bookmia_25_book.jsonl")
    bookname_snippets_indexs = merge_bookname(source_data)
    if llm == 'llama-7b':
        llm = "huggyllama/llama-7b"
    else:
        llm = "meta-llama/Llama-2-7b"
    if feature == "loss":
        feature = "losses"
    else:
        feature = "perplexities"

    features = np.load(f"results/{llm}/{feature}.npy")

    scores = np.array([item[0] - np.mean(item[1:]) for item in features])

    book_gt_labels = []
    X_scores = []

    for book_name, snippets_indexs in bookname_snippets_indexs.items():
        book_gt_labels.append(source_data[snippets_indexs[0]]['label'])
        tmp_uc_metrics = np.min(scores[snippets_indexs], axis=0)
        X_scores.append(tmp_uc_metrics)

    thr, preds = find_best_threshold(X_scores, book_gt_labels)
    outliers = preds

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
