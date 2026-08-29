#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: DirtyBoy
@Date: 2026/8/15 10:09
"""
import json


def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


if __name__ == '__main__':
    import numpy as np

    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
    )

    L = 64
    # =========================
    # 1. Load features
    # =========================
    data = read_from_jsonl(f"Answer/wiki_{L}_features_7d.jsonl")

    print(f"Number of samples: {len(data)}")
    print("Example:", data[0])

    # =========================
    # 2. Construct X and y
    # =========================
    feature_names = [
        "lexical_similarity",
        "num_sem_sets",
        "eigval_laplacian",
        "eccentricity",
        "degmat",
    ]

    X = np.array([
        [item["uc-metrics"][name] for name in feature_names]
        for item in data
    ], dtype=np.float64)

    y_true = np.array([
        item["label"]
        for item in data
    ], dtype=int)

    print("X shape:", X.shape)
    print("y shape:", y_true.shape)

    # =========================
    # 3. Standardization
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # 4. GMM clustering
    # =========================
    cluster = GaussianMixture(
        n_components=2,
        # covariance_type="full",
        random_state=42,
    )

    y_cluster = cluster.fit_predict(X_scaled)

    # =========================
    # 5. Align cluster IDs with labels
    # =========================

    acc_original = accuracy_score(y_true, y_cluster)

    y_flipped = 1 - y_cluster
    acc_flipped = accuracy_score(y_true, y_flipped)

    if acc_flipped > acc_original:
        y_pred = y_flipped
    else:
        y_pred = y_cluster

    # =========================
    # 6. Evaluation
    # =========================
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

    MSG = "The accuracy on the dataset is {:.2f}%"
    print(MSG.format(accuracy * 100))
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    MSG = "The balanced accuracy on the dataset is {:.2f}%"
    print(MSG.format(balanced_accuracy * 100))
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
