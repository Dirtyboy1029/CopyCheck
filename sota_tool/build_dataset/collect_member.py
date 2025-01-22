# -*- coding: utf-8 -*- 
# @Time : 2024/12/18 21:06 
# 
# @File : build_json_for_uncertainty_dataset.py
from datasets import load_from_disk
import numpy as np
from datasets import Dataset

dataset_dir = "/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/Database/Project_Gutenberg_20190210/train"
#

tokenized_dataset = load_from_disk(dataset_dir)

print(tokenized_dataset.column_names)


def transform_example(example, idx):
    return {
        'book': example['book_title'],  
        'bookid': idx + 10000, 
        'text': example['text'],  
        'label': 1  
    }


transformed_dataset = tokenized_dataset.map(transform_example, with_indices=True)
print(transformed_dataset.column_names)
print(transformed_dataset[:3])
output_dir = '/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/Database/pg19_gutenberg/member'

transformed_dataset = transformed_dataset.remove_columns(
    ['book_title', 'original_publication_year', 'gutenberg_release_date'])

print(transformed_dataset.column_names)

transformed_dataset.save_to_disk(output_dir)
print(transformed_dataset[:1]['book'])
