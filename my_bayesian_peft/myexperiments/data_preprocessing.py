# -*- coding: utf-8 -*- 
# @Time : 2023/12/12 21:05 
# @Author : DirtyBoy 
# @File : data_preprocessing.py
import json, random
import numpy as np
from utils import read_from_jsonl, write_to_jsonl
from sklearn.metrics import accuracy_score
from itertools import groupby
from operator import itemgetter


def dump_joblib(data, path):
    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        print('save conf file to ', path)
        return
    except IOError:
        raise IOError("Dump data failed.")


def modify_label(source_data, modify_type, ratio):
    gt_label = np.array([item['label'] for item in source_data])
    all_index = list(range(len(source_data)))
    if modify_type == 'random':
        noise_index = random.sample(all_index, k=int(len(source_data) * ratio / 100))
    elif modify_type == 'onlyseen':
        unseen_index = list(np.where(gt_label == 0)[0])
        noise_index = random.sample(unseen_index, k=int(len(unseen_index) * ratio / 100))
    else:
        all_index = []
        noise_index = []
    noise_label = []
    for index in all_index:
        if index in noise_index:
            noise_label.append((gt_label[index] + 1) % 2)
        else:
            noise_label.append(gt_label[index])
    for i, item in enumerate(source_data):
        item['label'] = int(noise_label[i])
    return source_data


def collect_balance_data(source_data, ratio):
    print('**************build trainset********************')
    print('source dataset contains ', len(source_data), ' samples')
    gt_label = np.array([item['label'] for item in source_data])
    print('contain ', int(sum(gt_label)), ' seen samples')
    print('contain ', len(gt_label) - int(sum(gt_label)), ' unseen samples')
    source_data.sort(key=itemgetter('book_id'))
    grouped_data = {k: list(v) for k, v in groupby(source_data, key=itemgetter('book_id'))}
    print('------------------------------------------------')
    print('There are ', len(grouped_data), ' books')
    seen_id = [book_id for book_id, snippets in grouped_data.items() if snippets[0]['label'] == 1]
    unseen_id = [book_id for book_id, snippets in grouped_data.items() if snippets[0]['label'] == 0]
    print('seen books: ', len(seen_id))
    print('unseen books:', len(unseen_id))
    print('------------------------------------------------')
    seen_id = random.sample(seen_id, k=50 - ratio)
    noise_id = random.sample(unseen_id, k=ratio)
    unselected_unseen_id = [item for item in unseen_id if item not in noise_id]
    unseen_id = random.sample(unselected_unseen_id, k=50)

    seen_data = []
    for item in seen_id:
        seen_data = seen_data + grouped_data[item]
    noise_data = []
    for item in noise_id:
        noise_data = noise_data + grouped_data[item]
    unseen_data = []
    for item in unseen_id:
        unseen_data = unseen_data + grouped_data[item]
    data = seen_data + unseen_data + noise_data
    gt_label = np.array([item['label'] for item in data])
    for item in noise_data:
        item['label'] = 1
    noise_label = np.array([item['label'] for item in data])
    random.shuffle(data)
    print(accuracy_score(gt_label, noise_label))
    return data


def collect_both_data(source_data, base_seen_ratio, unseen_ratio):
    print('**************build trainset********************')
    print('source dataset contains ', len(source_data), ' samples')
    gt_label = np.array([item['label'] for item in source_data])
    print('contain ', int(sum(gt_label)), ' seen samples')
    print('contain ', len(gt_label) - int(sum(gt_label)), ' unseen samples')
    source_data.sort(key=itemgetter('book_id'))
    grouped_data = {k: list(v) for k, v in groupby(source_data, key=itemgetter('book_id'))}
    print('------------------------------------------------')
    print('There are ', len(grouped_data), ' books')
    seen_id = [book_id for book_id, snippets in grouped_data.items() if snippets[0]['label'] == 1]
    all_unseen_id = [book_id for book_id, snippets in grouped_data.items() if snippets[0]['label'] == 0]
    print('seen books: ', len(seen_id))
    print('unseen books:', len(all_unseen_id))
    print('------------------------------------------------')
    base_seen_num = int(base_seen_ratio / 2)
    base_seen_id = random.sample(seen_id, k=base_seen_num)
    unseen_id = random.sample(all_unseen_id, k=50)
    noise_seen_num = int((50 - base_seen_num) * unseen_ratio / 100)
    noise_seen_id = random.sample([item for item in all_unseen_id if item not in unseen_id],
                                  k=noise_seen_num)
    seen_id = random.sample([item for item in seen_id if item not in base_seen_id],
                            k=50 - base_seen_num - noise_seen_num)
    id_dict = {'unseen_id': unseen_id,
               'base_seen_id': base_seen_id,
               'noise_seen_id': noise_seen_id,
               'seen_id': seen_id}
    dump_joblib(id_dict, f'../database/config/bookid_{base_seen_ratio}_{unseen_ratio}.conf')
    unseen_data = []
    for item in unseen_id:
        unseen_data = unseen_data + grouped_data[item]
    seen_data = []
    for item in seen_id:
        seen_data = seen_data + grouped_data[item]
    base_seen_data = []
    for item in base_seen_id:
        base_seen_data = base_seen_data + grouped_data[item]
    noise_seen_data = []
    for item in noise_seen_id:
        noise_seen_data = noise_seen_data + grouped_data[item]
    data = seen_data + unseen_data + base_seen_data + noise_seen_data
    gt_label = np.array([item['label'] for item in data])
    for item in noise_seen_data:
        item['label'] = 1
    data = seen_data + unseen_data + base_seen_data + noise_seen_data
    noise_label = np.array([item['label'] for item in data])
    random.shuffle(data)
    print(accuracy_score(gt_label, noise_label))
    return data


if __name__ == '__main__':

    ## base_seen_ratio  10   50   90

    data_type = 'both'  ## both banlance
    if data_type == 'unbanlance':
        modify_type = 'onlyseen'
        ratio = 90
        data = read_from_jsonl(file_path='../database/bookmia.jsonl')
        save_path = '../database/bookmia_' + modify_type + '_' + str(ratio) + '.jsonl'
        tmp_data = modify_label(data, modify_type=modify_type, ratio=ratio)
        write_to_jsonl(data=tmp_data, file_path=save_path)
    elif data_type == 'banlance':
        modify_type = 'onlyseen'
        ratio = 45
        data = read_from_jsonl(file_path='../database/my_bookmia.jsonl')
        save_path = '../database/bookmia_b' + modify_type + '_' + str(ratio) + '.jsonl'
        tmp_data = collect_balance_data(data, ratio=ratio)
        write_to_jsonl(data=tmp_data, file_path=save_path)
    elif data_type == 'both':
        # for base_seen_ratio in [10, 50, 90]:
        # for unseen_ratio in [20, 40, 60, 80]:
        base_seen_ratio = 10
        unseen_ratio = 90
        modify_type = data_type
        source_data = read_from_jsonl(file_path='../database/my_bookmia.jsonl')
        save_path = f'../database/bookmia_{modify_type}{base_seen_ratio}_{unseen_ratio}.jsonl'
        tmp_data = collect_both_data(source_data, base_seen_ratio, unseen_ratio)
        write_to_jsonl(data=tmp_data, file_path=save_path)


