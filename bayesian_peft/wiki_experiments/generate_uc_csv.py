# -*- coding: utf-8 -*- 
# @Time : 2025/5/12 16:02 
# @Author : DirtyBoy 
# @File : generate_uc_csv.py
import os, json
import numpy as np
from scipy.special import softmax
from metrics_utils import *
import pandas as pd


def count_2d_arrays(lst):
    count = 0
    for item in lst:
        if isinstance(item, np.ndarray) and item.ndim == 2:
            count += 1
        else:
            break
    return count



def build_save_path(save_folder):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)


def read_local_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def main_(llm_type, model_type, noise_ratio, epoch):
    if model_type == 'deepensemble':
        model_num = 10
        model_num_ = 1
    else:
        model_num = 1
        model_num_ = 10

    source_data = read_local_file(f'../database/wiki_bonlyseen_{noise_ratio}.jsonl')
    noise_labels = [item['label'] for item in source_data]

    conf = read_joblib(f'../database/config/wiki_id_{noise_ratio}.conf')
    seen_page_id = conf['seen_id']

    gt_labels = [1 if item['page_id'] in seen_page_id else 0 for item in source_data]

    sample_num = len(read_joblib(
        f'output/{model_type}/wiki_bonlyseen_{noise_ratio}_epoch{epoch}/{llm_type}-{model_num}_1.data'))

    soft_prob = np.zeros((10, sample_num, 2))
    prob = np.zeros((10, sample_num))

    for model_index in range(model_num_):
        data = np.array(read_joblib(
            f'output/{model_type}/wiki_bonlyseen_{noise_ratio}_epoch{epoch}/{llm_type}-{model_num}_{model_index + 1}.data'))
        if model_type == 'deepensemble':
            data = np.transpose(data, (1, 0, 2))
            for index in range(len(data)):
                softmax_data = softmax(data[index], axis=1)
                soft_prob[index] = softmax_data
                soft_prob_ = [item[1] for item in softmax_data]
                prob[index] = np.array(soft_prob_)
        else:
            softmax_data = softmax(data, axis=1)
            soft_prob[model_index] = softmax_data
            soft_prob_ = [item[1] for item in softmax_data]
            prob[model_index] = np.array(soft_prob_)
    my_prob = prob.T
    my_prob_mean = np.mean(my_prob, axis=1)
    soft_prob = np.transpose(soft_prob, (1, 0, 2))
    aleatoric_uc_set = []
    epistemic_uc_set = []
    for item in soft_prob:
        aleatoric_uc_, epistemic_uc_ = compute_variance(item)
        aleatoric_uc_set.append(aleatoric_uc_)
        epistemic_uc_set.append(epistemic_uc_)

    prob_set = np.array([np.mean(item) for item in my_prob])
    entropy_set = np.array([entropy(item, number=10) for item in my_prob])
    kld_set = np.array([predictive_kld(item, number=10) for item in my_prob])
    std_set = np.array([predictive_std(item, number=10) for item in my_prob])
    nll_set = np.array([nll(item, label=noise_labels[i]) for i, item in enumerate(my_prob)])
    wd_set = np.array([Wasserstein_distance(item, label=noise_labels[i]) for i, item in enumerate(my_prob)])
    ed_set = np.array([Euclidean_distance(item, label=noise_labels[i]) for i, item in enumerate(my_prob)])
    cd_set = np.array([Chebyshev_distance(item, label=noise_labels[i]) for i, item in enumerate(my_prob)])
    kdp_set = np.array([prob_label_kld(item, label=noise_labels[i]) for i, item in enumerate(my_prob)])
    max_set = np.array([max_max2(item) for item in my_prob])
    min_set = np.array([min2_min(item) for item in my_prob])
    mean_med_set = np.array([mean_med(item) for item in my_prob])


    uc_metrics_dict = {'gt_label': gt_labels[:len(prob_set)],
                       'noise_label': noise_labels[:len(prob_set)],
                       'prob': prob_set,
                       'predictive_entropy': entropy_set,
                       'predictive_kld': kld_set,
                       'predictive_std': std_set,
                       'max_set': max_set,
                       'min_set': min_set,
                       'mean_med_set': mean_med_set,
                       'nll_set': nll_set,
                       'wd_set': wd_set, 'ed_set': ed_set, 'cd_set': cd_set, 'kdp_set': kdp_set,
                       'aleatoric_uc': aleatoric_uc_set,
                       'epistemic_uc': epistemic_uc_set}
    for k,v in uc_metrics_dict.items():
        print(len(v))
    build_save_path(save_folder='csv_folder')
    save_folder = 'csv_folder'
    pd.DataFrame(uc_metrics_dict).to_csv(
        os.path.join(save_folder, f'wiki_uc_feature_csv_{llm_type}_{model_type}_{noise_ratio}_{epoch}.csv'))


if __name__ == '__main__':
    llm_type = 'roberta-large'
    for model_type in ['blob','mcdropout','deepensemble']:
        for noise_ratio in [20]:
            main_(llm_type, model_type, noise_ratio, epoch=3)
