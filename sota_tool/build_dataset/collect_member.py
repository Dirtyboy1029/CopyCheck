# -*- coding: utf-8 -*- 
# @Time : 2024/12/18 21:06 
# @Author : DirtyBoy 
# @File : build_json_for_uncertainty_dataset.py
from datasets import load_from_disk
import numpy as np
from datasets import Dataset
# # 设定保存数据集的目录
dataset_dir = "/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/Database/Project_Gutenberg_20190210/train"
#
# # 加载数据集
tokenized_dataset = load_from_disk(dataset_dir)

print(tokenized_dataset.column_names)


def transform_example(example, idx):
    return {
        'book': example['book_title'],  # 将 'book_title' 改为 'book'
        'bookid': idx + 10000,  # 给每本书添加一个顺序编号作为 'bookid'
        'text': example['text'],  # 保持 'text' 字段不变
        'label': 1  # 设置 'label' 为 1
    }


transformed_dataset = tokenized_dataset.map(transform_example, with_indices=True)
print(transformed_dataset.column_names)
print(transformed_dataset[:3])
output_dir = '/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/Database/pg19_gutenberg/member'
# 选择需要的列
transformed_dataset = transformed_dataset.remove_columns(
    ['book_title', 'original_publication_year', 'gutenberg_release_date'])

print(transformed_dataset.column_names)

transformed_dataset.save_to_disk(output_dir)
print(transformed_dataset[:1]['book'])