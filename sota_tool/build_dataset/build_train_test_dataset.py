# -*- coding: utf-8 -*- 
# @Time : 2024/12/20 10:30 
# @Author : DirtyBoy 
# @File : build_train_test_dataset.py
from datasets import load_from_disk
import random, os, re
from datasets import Dataset
from datasets import concatenate_datasets

member = load_from_disk('/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/Database/pg19_gutenberg/member')
non_member = load_from_disk(
    '/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/Database/pg19_gutenberg/non_member')


def preprocess_text(text):
    return re.sub(r'\n+', '\n', text)


def split_text_into_segments(text, min_words=512):
    # 假设 preprocess_text 是用来预处理文本的函数
    text = preprocess_text(text)

    # 用正则表达式提取所有有效单词
    words = re.findall(r'\b\w+\b', text)  # \b 是单词边界，\w+ 匹配一个或多个字母数字字符

    segments = []
    segment = []
    word_count = 0

    for word in words:
        segment.append(word)
        word_count += 1

        if word_count >= min_words:
            segments.append(' '.join(segment))  # 用空格连接单词并保存
            segment = []  # 清空当前片段
            word_count = 0  # 重置单词计数

    # 如果最后还有剩余的单词，添加到结果中
    if segment:
        segments.append(' '.join(segment))

    return segments




def count_long_text_samples(dataset):
    long_text_samples = dataset.filter(
        lambda example: len(split_text_into_segments(example['text'])) > 95 )
    return long_text_samples


long_text_member = count_long_text_samples(member)
print(len(long_text_member))
long_text_non_member = count_long_text_samples(non_member)
print(len(long_text_non_member))

def random_select(train, k=1200):
    train_list = list(train)
    random_samples = random.sample(train_list, k)
    return Dataset.from_dict({
        key: [example[key] for example in random_samples]  # 每个字段值作为列表
        for key in train_list[0].keys()
    })


def convert_to_single_value(example):
    # 将字典中的每个字段值从列表转换为单一值
    for key in example:
        if isinstance(example[key], list) and len(example[key]) == 1:
            example[key] = example[key][0]  # 取列表中的第一个元素
    return example


def a(long_text_member):
    data_list = list(long_text_member)
    modified_data = [convert_to_single_value(example) for example in data_list]
    long_text_member = Dataset.from_dict({
        key: [example[key] for example in modified_data] for key in modified_data[0].keys()
    })
    return long_text_member


long_text_member = random_select(long_text_member, 1100)
long_text_member = a(long_text_member)
long_text_non_member = random_select(long_text_non_member, 1150)
long_text_non_member = a(long_text_non_member)

print(long_text_member[:1])
train_member_set = long_text_member.select(range(1000))
test_member_set1 = long_text_member.select(range(1000, 1010))
test_member_set2 = long_text_member.select(range(1010, 1030))
test_member_set3 = long_text_member.select(range(1030, 1060))
test_member_set4 = long_text_member.select(range(1060, 1100))

train_non_member_set = long_text_non_member.select(range(1000))
test_non_member_set1 = long_text_non_member.select(range(1000, 1040))
test_non_member_set2 = long_text_non_member.select(range(1040, 1070))
test_non_member_set3 = long_text_non_member.select(range(1070, 1090))
test_non_member_set4 = long_text_non_member.select(range(1090, 1100))

merged_train_dataset = concatenate_datasets([train_member_set, train_non_member_set])
merged_dataset_dir = "/opt/data/private/LHD_LLM/LLM_uncertainty/document-level-membership-inference/data/comparative_experiments_dataset/my_exp_trainset"
if not os.path.isdir(merged_dataset_dir):
    os.makedirs(merged_dataset_dir)

merged_train_dataset.save_to_disk(merged_dataset_dir)

merged_test_dataset1 = concatenate_datasets([test_member_set1, test_non_member_set1])
print(len(merged_test_dataset1))
merged_test_dataset2 = concatenate_datasets([test_member_set2, test_non_member_set2])
print(len(merged_test_dataset2))
merged_test_dataset3 = concatenate_datasets([test_member_set3, test_non_member_set3])
print(len(merged_test_dataset3))
merged_test_dataset4 = concatenate_datasets([test_member_set4, test_non_member_set4])
print(len(merged_test_dataset4))

merged_testset_dir1 = "/opt/data/private/LHD_LLM/LLM_uncertainty/document-level-membership-inference/data/comparative_experiments_dataset/my_exp_testset1"
if not os.path.isdir(merged_testset_dir1):
    os.makedirs(merged_testset_dir1)
merged_test_dataset1.save_to_disk(merged_testset_dir1)

merged_testset_dir2 = "/opt/data/private/LHD_LLM/LLM_uncertainty/document-level-membership-inference/data/comparative_experiments_dataset/my_exp_testset2"
if not os.path.isdir(merged_testset_dir2):
    os.makedirs(merged_testset_dir2)
merged_test_dataset2.save_to_disk(merged_testset_dir2)

merged_testset_dir3 = "/opt/data/private/LHD_LLM/LLM_uncertainty/document-level-membership-inference/data/comparative_experiments_dataset/my_exp_testset3"
if not os.path.isdir(merged_testset_dir3):
    os.makedirs(merged_testset_dir3)
merged_test_dataset3.save_to_disk(merged_testset_dir3)

merged_testset_dir4 = "/opt/data/private/LHD_LLM/LLM_uncertainty/document-level-membership-inference/data/comparative_experiments_dataset/my_exp_testset4"
if not os.path.isdir(merged_testset_dir4):
    os.makedirs(merged_testset_dir4)
merged_test_dataset4.save_to_disk(merged_testset_dir4)

unseen_set = long_text_non_member.select(range(1100, 1150))

merged_unseen_set_dir = "/opt/data/private/LHD_LLM/LLM_uncertainty/document-level-membership-inference/data/comparative_experiments_dataset/my_exp_unseenset"
if not os.path.isdir(merged_unseen_set_dir):
    os.makedirs(merged_unseen_set_dir)
unseen_set.save_to_disk(merged_unseen_set_dir)
