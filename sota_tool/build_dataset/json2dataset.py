# -*- coding: utf-8 -*- 
# @Time : 2025/1/20 23:34 
# @Author : DirtyBoy 
# @File : json2dataset.py
from datasets import Dataset
from datasets import load_from_disk
import json, re, os
import numpy as np
from transformers import LlamaTokenizer
from tqdm import tqdm
from collections import defaultdict


def merge_bookname(data):
    merged_data = defaultdict(list)
    for item in data:
        merged_data[item['book']].append(item)
    return dict(merged_data)


def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def main_(name='bonlyseen_10'):
    source_data = read_from_jsonl(
        file_path='/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/database/my_bookmia.jsonl')
    my_data = read_from_jsonl(
        '/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/database/bookmia_' + name + '.jsonl')

    noise_label = np.array([item['label'] for item in my_data])
    suspect_seen_index = np.where(noise_label == 1)[0]
    goal_book = set([my_data[index]['book'] for index in suspect_seen_index])

    snippet_sets = merge_bookname(my_data)
    source_book = merge_bookname(source_data)
    dataset = []
    labels = []
    for i, item in enumerate(tqdm(goal_book, desc='merge data')):
        data = snippet_sets[item]
        book_text = ''
        sorted_snippets = sorted(data, key=lambda x: x['snippet_id'])
        restored_text = ' '.join(snippet['snippet'] for snippet in sorted_snippets)
        dataset.append({
            'book': data[0]['book'],
            'bookid': i,
            'text': restored_text,
            'label': source_book[item][0]['label']
        })
        labels.append(source_book[item][0]['label'])

    dataset = Dataset.from_list(dataset)

    dataset.save_to_disk(
        "/opt/data/private/LHD_LLM/LLM_uncertainty/document-level-membership-inference/data/comparative_experiments_dataset/" + name)
    print(name)
    print(len(labels))
    print(sum(labels))


if __name__ == '__main__':
    for name in ['bonlyseen_10', 'bonlyseen_20', 'bonlyseen_30', 'bonlyseen_40']:
        main_(name)
