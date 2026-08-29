#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: DirtyBoy
@Date: 2026/7/25 12:30
"""
import os, json, argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def pca(data, n_components=10):
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    data_standardized = np.nan_to_num(data_standardized)
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data_standardized)
    return reduced_data


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


if __name__ == '__main__':
    L = 300
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_type', '-dt', type=str, default='book')
    parser.add_argument('-detection_algorithm_type', '-dat', type=str, default='gmm')
    #  if dbscan  kmeans  hierarchical  svm  gmm
    parser.add_argument('-noise_ratio', '-nr', type=str, default='25')
    parser.add_argument('-llm_type', '-lm', type=str, default="meta-llama/Llama-2-13b")
    #  'huggyllama/llama-7b'  "meta-llama/Llama-2-7b"  "EleutherAI/gpt-j-6b"  "facebook/opt-6.7b" "Qwen/Qwen1.5-7B"
    args = parser.parse_args()
    llm_type = args.llm_type
    detection_algorithm_type = args.detection_algorithm_type
    data_type = args.data_type
    noise_ratio = args.noise_ratio
    aa = 0
    if aa == 0:
        A = 0
        B = 1
    else:
        A = 1
        B = 0

    data_name = f"bookmia_{noise_ratio}_{data_type}"
    data = read_from_jsonl(f'../Database/{data_name}.jsonl')
    data = data[0:int(len(data) / 2)]
    suspect_labels = np.array([item['target'] for item in data])
    goal_indexs = np.where(suspect_labels == 1)[0]

    data = [item for i, item in enumerate(data) if i in goal_indexs]

    bookname_snippets_indexs = merge_bookname(data, data_type=data_type)
    book_names = list(bookname_snippets_indexs.keys())

    # uc_metrics = pd.read_csv(f'uc_metrics_csv/{llm_type}/{data_name}.csv').to_numpy()[goal_indexs]
    df = pd.read_csv(f'uc_metrics_csv_{L}/{llm_type}/{data_name}.csv')
    keywords = ["deep", "mc", "blob"]  # ,"deep"
    cols = [col for col in df.columns if any(k in col for k in keywords)]
    df_filtered = df[cols]
    uc_metrics = df_filtered.to_numpy()[goal_indexs]

    book_gt_labels = []
    X_features = []

    for book_name, snippets_indexs in bookname_snippets_indexs.items():
        book_gt_labels.append(data[snippets_indexs[0]]['label'])
        tmp_uc_metrics = np.max(uc_metrics[snippets_indexs], axis=0)
        X_features.append(tmp_uc_metrics)

    X_feature = np.array(X_features)
    X_feature = pca(X_feature)

    if detection_algorithm_type == 'if':
        from sklearn.ensemble import IsolationForest

        isolation_forest = IsolationForest(n_estimators=50, contamination=0.2)
        isolation_forest.fit(X_feature)
        outliers = isolation_forest.predict(X_feature)

    elif detection_algorithm_type == 'kmeans':
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=2, random_state=4321)
        kmeans.fit(X_feature)
        outliers = kmeans.labels_

    elif detection_algorithm_type == 'gmm':
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X_feature)
        outliers = gmm.predict(X_feature)

    elif detection_algorithm_type == 'svm':
        from sklearn.svm import OneClassSVM

        ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        ocsvm.fit(X_feature)

        outliers = ocsvm.predict(X_feature)

    elif detection_algorithm_type == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering

        hier_cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
        outliers = hier_cluster.fit_predict(X_feature)

    elif detection_algorithm_type == 'dbscan':

        from sklearn.cluster import DBSCAN

        dbscan = DBSCAN()
        dbscan.fit(X_feature)
        outliers = dbscan.labels_

    else:
        outliers = None


    def new_label(outliers, X_feature, detection_algorithm_type):
        if detection_algorithm_type == 'gmm' or detection_algorithm_type == 'kmeans' or detection_algorithm_type == 'hierarchical':
            a_l = 1
            b_l = 0
        elif detection_algorithm_type == 'if' or detection_algorithm_type == 'svm':
            a_l = 1
            b_l = -1
        else:
            a_l = -1
            b_l = 0

        a_index = np.where(outliers == a_l)[0]
        b_index = np.where(outliers == b_l)[0]
        a_uc = np.mean(X_feature[a_index])
        b_uc = np.mean(X_feature[b_index])
        if b_uc > a_uc:
            b_label = A
            a_label = B
        else:
            b_label = B
            a_label = A
        new_outliers = np.zeros(X_feature.shape[0])
        new_outliers[a_index] = a_label
        new_outliers[b_index] = b_label
        return new_outliers


    outliers = new_label(outliers, X_feature, detection_algorithm_type)
    print(outliers)
    book_gt_labels = np.array(book_gt_labels)
    for i in np.where((outliers == 0) & (book_gt_labels == 1))[0]:
        print(book_names[i])
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
