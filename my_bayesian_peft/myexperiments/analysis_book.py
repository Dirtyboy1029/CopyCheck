# -*- coding: utf-8 -*- 
# @Time : 2024/11/26 20:55 
# 
# @File : analysis_book.py
from utils import read_local_file, read_joblib
import numpy as np
from collections import defaultdict
import os
import matplotlib.pyplot as plt


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


def analysis_book_onlyseen(detection_algorithm_type='dbscan', noise_ratio=30):
    noise_type = 'bonlyseen'
    data_type = 'bookmia_' + noise_type + '_' + str(noise_ratio)
    noise_data = read_local_file(file_path='../database/' + data_type + '.jsonl')
    data = read_local_file(file_path='../database/my_bookmia.jsonl')
    data_dict = {(item['book_id'], item['snippet_id']): item['label'] for item in data}

    # gt_label = [data_dict[(item['book_id'], item['snippet_id'])]
    #             for item in noise_data
    #             if (item['book_id'], item['snippet_id']) in data_dict]
    # gt_label = np.array(gt_label)[0:(len(gt_label) // 4) * 4]
    noise_label = np.array([item['label'] for item in noise_data])[0:(len(noise_data) // 4) * 4]

    suspect_seen_index = np.where(noise_label == 1)[0]
    goal_book = set([noise_data[index]['book'] for index in suspect_seen_index])
    label_error_mask = np.load(os.path.join('label_error_masks/anomaly_detection',
                                            f'{detection_algorithm_type}_{noise_type}_{noise_ratio}.npy'))
    print(label_error_mask)
    for i, index in enumerate(suspect_seen_index):
        if label_error_mask[i] == -1:
            noise_data[index]['label'] = 0
    source_book = merge_bookname(data)
    detection_book = merge_bookname(noise_data)

    seen = {}
    unseen = {}
    for i, (key, values) in enumerate(detection_book.items()):
        if key in goal_book:
            print('--------------------------------')
            print(i + 1, key)
            print(source_book[key][0]['label'])
            if source_book[key][0]['label'] == 1:
                seen[key] = [sum([item['label'] for item in values]), len(values)]
            else:
                unseen[key] = [sum([item['label'] for item in values]), len(values)]
            print(sum([item['label'] for item in values]), len(values))
    print(len(seen), len(unseen))

    dict1 = seen
    dict2 = unseen
    sorted_dict1 = dict(sorted(dict1.items(), key=lambda x: x[1][0], reverse=True))
    sorted_dict2 = dict(sorted(dict2.items(), key=lambda x: x[1][0], reverse=True))

    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].bar(sorted_dict1.keys(), [v[0] for v in sorted_dict1.values()], color='blue', alpha=0.7)
    axes[0].set_xlabel("books")
    axes[0].set_ylabel("number of  samples(model predict seen) ")
    axes[0].set_ylim(0, 100)

    axes[1].bar(sorted_dict2.keys(), [v[0] for v in sorted_dict2.values()], color='green', alpha=0.7)
    axes[1].set_xlabel("books")
    axes[1].set_ylabel("number of  samples(model predict seen) ")
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    plt.show()

    book_name = 'treasure island.txt'  # Tress_of_the_Emerald_Sea.txt Said_No_One_Ever.txt
    goal = detection_book[book_name]
    id = []
    for item in goal:
        if item['label'] == 1:
            id.append(item['snippet_id'])
    print(sorted(id))


def analysis_book_both(detection_algorithm_type='dbscan', noise_ratio=30, noise_type='both10'):
    data_type = f'bookmia_{noise_type}_{noise_ratio}'
    noise_data = read_local_file(file_path=f'../database/{data_type}.jsonl')
    noise_label = np.array([item['label'] for item in noise_data])[0:(len(noise_data) // 4) * 4]
    data = read_local_file(file_path='../database/my_bookmia.jsonl')
    tmp_noise_type = noise_type.replace('both', '')
    conf = read_joblib(f'../database/config/bookid_{tmp_noise_type}_{noise_ratio}.conf')
    suspect_seen_bookid = conf['noise_seen_id']
    seen_bookid = conf['seen_id']
    suspect_seen_index = get_samples_index(noise_data, suspect_seen_bookid)
    seen_index = get_samples_index(noise_data, seen_bookid)

    my_index = list(range(len(noise_label)))
    suspect_seen_index = [item for item in suspect_seen_index if item in my_index]
    seen_index = [item for item in seen_index if item in my_index]
    suspect_seen_book = set([noise_data[index]['book'] for index in suspect_seen_index])
    seen_book = set([noise_data[index]['book'] for index in seen_index])
    goal_book = suspect_seen_book.union(seen_book)
    label_error_mask = np.load(os.path.join('label_error_masks/binary_classification',
                                            f'{detection_algorithm_type}_{noise_type}_{noise_ratio}.npy'))
    for i, index in enumerate(seen_index + suspect_seen_index):
        if label_error_mask[i] == 0:
            noise_data[index]['label'] = 1
        else:
            noise_data[index]['label'] = 0
    source_book = merge_bookname(data)
    detection_book = merge_bookname(noise_data)
    seen = {}
    unseen = {}
    for i, (key, values) in enumerate(detection_book.items()):
        if key in goal_book:
            print('--------------------------------')
            print(i + 1, key)
            print(source_book[key][0]['label'])
            if source_book[key][0]['label'] == 1:
                seen[key] = [sum([item['label'] for item in values]), len(values)]
            else:
                unseen[key] = [sum([item['label'] for item in values]), len(values)]
            print(sum([item['label'] for item in values]), len(values))
    print(len(seen), len(unseen))

    dict1 = seen
    dict2 = unseen
    sorted_dict1 = dict(sorted(dict1.items(), key=lambda x: x[1][0], reverse=True))
    sorted_dict2 = dict(sorted(dict2.items(), key=lambda x: x[1][0], reverse=True))
    plt.style.use('default')
    x1 = np.arange(len(sorted_dict1))
    x2 = np.arange(len(sorted_dict2)) + len(sorted_dict1) + 0.1
    fig, ax = plt.subplots(figsize=(12, 6))
    print([v for v in sorted_dict2.values()])
    percent_seen = [v[0] / v[1] * 100 for v in sorted_dict1.values()]
    percent_unseen = [v[0] / v[1] * 100 for v in sorted_dict2.values()]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_seen = ax.bar(x1, percent_seen, width=0.7, color='blue', alpha=0.7, label='Seen')
    bars_unseen = ax.bar(x2, percent_unseen, width=0.7, color='green', alpha=0.7, label='Unseen')
    for bar, percent in zip(bars_seen, percent_seen):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{percent:.0f}', ha='center', va='bottom',
                fontsize=9)
    for bar, percent in zip(bars_unseen, percent_unseen):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{percent:.0f}', ha='center', va='bottom',
                fontsize=9)
    ax.set_xlabel("Books")
    ax.set_ylabel("Number of Samples (Model Predict Seen)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def analysis_book_new(detection_algorithm_type='svm', noise_ratio=10, noise_type='base10'):
    data_type = f'bookmia_{noise_type}_test{noise_ratio}'
    noise_data = read_local_file(file_path=f'../database/{data_type}.jsonl')
    noise_label = np.array([item['label'] for item in noise_data])[0:(len(noise_data) // 4) * 4]
    data = read_local_file(file_path='../database/my_bookmia.jsonl')
    conf = read_joblib(f'../database/config/bookid_{noise_type}_test{noise_ratio}.conf')
    unseen_bookid = conf['unseen_id']
    base_seen_bookid = conf['base_seen_id']
    base_unseen_bookid = conf['base_unseen_id']
    test_seen_bookid = conf['test_seen_id']
    test_unseen_bookid = conf['test_unseen_id']

    base_seen_id = get_samples_index(noise_data, base_seen_bookid)
    base_unseen_id = get_samples_index(noise_data, base_unseen_bookid)
    test_seen_id = get_samples_index(noise_data, test_seen_bookid)
    test_unseen_id = get_samples_index(noise_data, test_unseen_bookid)

    unseen_book = set([noise_data[index]['book'] for index in test_unseen_id])
    seen_book = set([noise_data[index]['book'] for index in test_seen_id])
    goal_book = unseen_book.union(seen_book)

    label_error_mask = np.load(os.path.join('label_error_masks/binary_classification',
                                            f'{detection_algorithm_type}_{noise_type}_{noise_ratio}.npy'))
    for i, index in enumerate(test_seen_id + test_unseen_id):
        if label_error_mask[i] == 0:
            noise_data[index]['label'] = 1
        else:
            noise_data[index]['label'] = 0
    source_book = merge_bookname(data)
    detection_book = merge_bookname(noise_data)
    seen = {}
    unseen = {}
    for i, (key, values) in enumerate(detection_book.items()):
        if key in goal_book:
            print('--------------------------------')
            print(i + 1, key)
            print(source_book[key][0]['label'])
            if source_book[key][0]['label'] == 1:
                seen[key] = [sum([item['label'] for item in values]), len(values)]
            else:
                unseen[key] = [sum([item['label'] for item in values]), len(values)]
            print(sum([item['label'] for item in values]), len(values))
    print(len(seen), len(unseen))

    dict1 = seen
    dict2 = unseen
    sorted_dict1 = dict(sorted(dict1.items(), key=lambda x: x[1][0], reverse=True))
    sorted_dict2 = dict(sorted(dict2.items(), key=lambda x: x[1][0], reverse=True))
    plt.style.use('default')
    x1 = np.arange(len(sorted_dict1))
    x2 = np.arange(len(sorted_dict2)) + len(sorted_dict1) + 0.1
    fig, ax = plt.subplots(figsize=(12, 6))
    print([v for v in sorted_dict2.values()])
    percent_seen = [v[0] / v[1] * 100 for v in sorted_dict1.values()]
    percent_unseen = [v[0] / v[1] * 100 for v in sorted_dict2.values()]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_seen = ax.bar(x1, percent_seen, width=0.7, color='blue', alpha=0.7, label='Seen')
    bars_unseen = ax.bar(x2, percent_unseen, width=0.7, color='green', alpha=0.7, label='Unseen')
    for bar, percent in zip(bars_seen, percent_seen):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{percent:.0f}', ha='center', va='bottom',
                fontsize=9)
    for bar, percent in zip(bars_unseen, percent_unseen):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{percent:.0f}', ha='center', va='bottom',
                fontsize=9)
    ax.set_xlabel("Books")
    ax.set_ylabel("Number of Samples (Model Predict Seen)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #analysis_book_onlyseen(detection_algorithm_type='if', noise_ratio=50)
    # analysis_book_both(detection_algorithm_type='svm', noise_ratio=40, noise_type='both10')
    analysis_book_new(detection_algorithm_type='svm', noise_ratio=10, noise_type='base10')
