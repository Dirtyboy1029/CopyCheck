# -*- coding: utf-8 -*- 
# @Time : 2024/12/17 19:16 
# 
# @File : unseen_books_detection.py

from utils import read_local_file, read_joblib
import numpy as np
from collections import defaultdict
import os


def get_samples_index(source_data, book_ids):
    my_len = len(source_data) - (len(source_data) % 4)

    tmp_id = []
    for id in book_ids:
        for i, item in enumerate(source_data):
            if item['book_id'] == id and i < my_len:
                tmp_id.append(i)
    return tmp_id


def merge_bookname(data):
    merged_data = defaultdict(list)
    for item in data:
        merged_data[item['book']].append(item)
    return dict(merged_data)


def detect_onlyseen_book(thr=0.6,
                         llm_type='llama-7b',
                         noise_type='bonlyseen',
                         noise_ratio='10',
                         detection_algorithm_type='dbscan'):
    source_bookmia_data = read_local_file(file_path='../database/my_bookmia.jsonl')
    noise_bookmia_data = read_local_file(file_path=f'../database/bookmia_{noise_type}_{noise_ratio}.jsonl')
    noise_label = np.array([item['label'] for item in noise_bookmia_data])[0:(len(noise_bookmia_data) // 4) * 4]
    suspect_seen_index = np.where(noise_label == 1)[0]
    suspect_book = set([noise_bookmia_data[index]['book'] for index in suspect_seen_index])
    source_book = merge_bookname(source_bookmia_data)
    seen_book = []
    unseen_book = []
    for item in suspect_book:
        if source_book[item][0]['label'] == 1:
            seen_book.append(item)
        else:
            unseen_book.append(item)
    label_error_mask = np.load(os.path.join(f'label_error_masks/anomaly_detection/{llm_type}',
                                            f'{detection_algorithm_type}_{noise_type}_{noise_ratio}.npy'))
    print(list(label_error_mask))
    print(np.sum(label_error_mask))
    for i, index in enumerate(suspect_seen_index):
        if label_error_mask[i] != 0:
            noise_bookmia_data[index]['label'] = 0
    detection_book = merge_bookname(noise_bookmia_data)
    gt_book_labels = list(np.zeros(len(seen_book))) + list(np.ones(len(unseen_book)))
    detection_book_labels = []
    for i, key in enumerate(seen_book + unseen_book):
        if sum([item['label'] for item in detection_book[key]]) / len(detection_book[key]) > thr:
            detection_book_labels.append(0)
        else:
            detection_book_labels.append(1)
    from sklearn.metrics import confusion_matrix, accuracy_score
    balanced_accuracy = accuracy_score(gt_book_labels, detection_book_labels)

    MSG = "The  accuracy on the dataset is {:.5f}%"
    print(MSG.format(balanced_accuracy * 100))
    tn, fp, fn, tp = confusion_matrix(gt_book_labels, detection_book_labels).ravel()
    print(f"True Negatives (tn): {tn} (Books correctly identified as 'unseen') / {tn + fp}")
    print(f"False Positives (fp): {fp} (Books incorrectly identified as 'unseen' but were 'seen') / {tn + fp}")
    print(f"False Negatives (fn): {fn} (Books incorrectly identified as 'seen' but were 'unseen') / {fn + tp}")
    print(f"True Positives (tp): {tp} (Books correctly identified as 'seen') / {fn + tp}")


def detect_base_book(thr=0.8,
                     llm_type='llama-7b',
                     noise_type='base10',
                     noise_ratio='10',
                     detection_algorithm_type='svm'):
    noise_bookmia_data = read_local_file(file_path=f'../database/bookmia_{noise_type}_test{noise_ratio}.jsonl')
    conf = read_joblib(f'../database/config/bookid_{noise_type}_test{noise_ratio}.conf')

    test_seen_bookid = conf['test_seen_id']
    test_unseen_bookid = conf['test_unseen_id']

    test_seen_id = get_samples_index(noise_bookmia_data, test_seen_bookid)
    test_unseen_id = get_samples_index(noise_bookmia_data, test_unseen_bookid)

    unseen_book = list(set([noise_bookmia_data[index]['book'] for index in test_unseen_id]))
    seen_book = list(set([noise_bookmia_data[index]['book'] for index in test_seen_id]))

    label_error_mask = np.load(os.path.join(f'label_error_masks/binary_classification/{llm_type}',
                                            f'{detection_algorithm_type}_{noise_type}_{noise_ratio}.npy'))

    for i, index in enumerate(test_seen_id + test_unseen_id):
        if label_error_mask[i] == 1:
            noise_bookmia_data[index]['label'] = 0
        else:
            noise_bookmia_data[index]['label'] = 1
    detection_book = merge_bookname(noise_bookmia_data)

    gt_book_labels = list(np.zeros(len(seen_book))) + list(np.ones(len(unseen_book)))
    detection_book_labels = []
    for i, key in enumerate(seen_book + unseen_book):
        if sum([item['label'] for item in detection_book[key]]) / len(detection_book[key]) > thr:
            detection_book_labels.append(0)
        else:
            detection_book_labels.append(1)
    from sklearn.metrics import confusion_matrix, accuracy_score
    balanced_accuracy = accuracy_score(gt_book_labels, detection_book_labels)

    MSG = "The accuracy on the dataset is {:.5f}%"

    tn, fp, fn, tp = confusion_matrix(gt_book_labels, detection_book_labels).ravel()
    print(f"using {detection_algorithm_type} to detection {noise_type} testset")
    print(f'There are {tn + fp + fn + tp} in testset')

    print(MSG.format(balanced_accuracy * 100))
    print(f"True Negatives (tn): {tn} (Books correctly identified as 'unseen') / {tn + fp}")
    print(f"False Positives (fp): {fp} (Books incorrectly identified as 'unseen' but were 'seen') / {tn + fp}")
    print(f"False Negatives (fn): {fn} (Books incorrectly identified as 'seen' but were 'unseen') / {fn + tp}")
    print(f"True Positives (tp): {tp} (Books correctly identified as 'seen') / {fn + tp}")

def detect_test_book(thr=0.6,
                         llm_type='open_llama_3b',
                         noise_type='test',
                         noise_ratio='seen10',
                         detection_algorithm_type='gmm'):

    noise_bookmia_data = read_local_file(file_path=f'../database/bookmia_{noise_type}_{noise_ratio}.jsonl')
    noise_label = np.array([item['label'] for item in noise_bookmia_data])[0:(len(noise_bookmia_data) // 4) * 4]
    suspect_seen_index = np.where(noise_label == 1)[0]
    suspect_book = set([noise_bookmia_data[index]['book'] for index in suspect_seen_index])
    suspect_books = merge_bookname(noise_bookmia_data)
    conf = read_joblib('../database/config/' + 'bookid_' + noise_type + '_' + str(noise_ratio) + '.conf')
    suspected_seen_ids = conf['suspected_seen_id']
    suspected_unseen_ids = conf['suspected_unseen_id']
    seen_book = []
    unseen_book = []
    for item in suspect_book:
        if suspect_books[item][0]['book_id'] in suspected_seen_ids:
            seen_book.append(item)
        else:
            unseen_book.append(item)
    label_error_mask = np.load(os.path.join(f'label_error_masks/anomaly_detection/{llm_type}',
                                            f'{detection_algorithm_type}_{noise_type}_{noise_ratio}.npy'))
    #print(list(label_error_mask))
    #print(np.sum(label_error_mask))
    for i, index in enumerate(suspect_seen_index):
        if label_error_mask[i] == 0:
            noise_bookmia_data[index]['label'] = 0
    detection_book = merge_bookname(noise_bookmia_data)
    gt_book_labels = list(np.ones(len(seen_book))) + list(np.zeros(len(unseen_book)))
    print(gt_book_labels)
    detection_book_labels = []
    for i, key in enumerate(seen_book + unseen_book):
        if sum([item['label'] for item in detection_book[key]]) / len(detection_book[key]) > thr:
            detection_book_labels.append(0)
        else:
            detection_book_labels.append(1)
    from sklearn.metrics import confusion_matrix, accuracy_score
    balanced_accuracy = accuracy_score(gt_book_labels, detection_book_labels)

    MSG = "The  accuracy on the dataset is {:.5f}%"
    print(MSG.format(balanced_accuracy * 100))
    tn, fp, fn, tp = confusion_matrix(gt_book_labels, detection_book_labels).ravel()
    print(f"True Negatives (tn): {tn} (Books correctly identified as 'unseen') / {tn + fp}")
    print(f"False Positives (fp): {fp} (Books incorrectly identified as 'unseen' but were 'seen') / {tn + fp}")
    print(f"False Negatives (fn): {fn} (Books incorrectly identified as 'seen' but were 'unseen') / {fn + tp}")
    print(f"True Positives (tp): {tp} (Books correctly identified as 'seen') / {fn + tp}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-noise_type', '-nt', type=str, default='bonlyseen')
    parser.add_argument('-detection_algorithm_type', '-dat', type=str, default='if')
    parser.add_argument('-noise_ratio', '-nr', type=str, default='40')
    parser.add_argument('-llm_type', '-lm', type=str, default='llama-7b')
    parser.add_argument('-thr', '-r', type=float, default=0.6)
    args = parser.parse_args()
    thr = args.thr
    llm_type = args.llm_type
    detection_algorithm_type = args.detection_algorithm_type
    noise_type = args.noise_type
    noise_ratio = args.noise_ratio

    if noise_type == 'bonlyseen':
        detect_onlyseen_book(thr=thr, llm_type=llm_type, noise_type=noise_type,
                             noise_ratio=noise_ratio, detection_algorithm_type=detection_algorithm_type)
    elif noise_type == 'test':
        detect_test_book(thr=thr, llm_type=llm_type, noise_type=noise_type,
                             noise_ratio=noise_ratio, detection_algorithm_type=detection_algorithm_type)
    else:
        detect_base_book(thr=thr, llm_type=llm_type, noise_type=noise_type,
                         noise_ratio=noise_ratio, detection_algorithm_type=detection_algorithm_type)
