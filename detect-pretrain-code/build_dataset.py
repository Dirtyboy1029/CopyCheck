#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: DirtyBoy
@Date: 2026/8/6 18:50
"""
import os, json

def write_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')

def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


my_data = []
data = read_from_jsonl('../Database/bookmia_25_arxiv818.jsonl')
print(data[0].keys())
# for item in data:
#     if item['target'] == 1:
#         my_data.append(item)
#
#
# write_to_jsonl(my_data, 'datasets/bookmia_25_book.jsonl')
