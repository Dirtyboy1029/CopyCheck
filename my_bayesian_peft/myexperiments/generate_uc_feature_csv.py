# -*- coding: utf-8 -*- 
# @Time : 2024/11/12 14:00 
# 
# @File : generate_uc_feature_csv.py
import pandas as pd
import os
from utils import read_local_file, collect_uc_metrics, read_joblib
from metrics_utils import compute_variance
import numpy as np


def build_save_path():
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)


def main_(noise_ratio=30,
          noise_type='bonlyseen', EPOCH=3, llm_type='llama-7b'):
    if noise_type == 'bonlyseen' or noise_type == 'test':
        Data_type = 'bookmia_' + noise_type + '_' + str(noise_ratio)
    else:
        Data_type = 'bookmia_' + noise_type + '_test' + str(noise_ratio)
    if not os.path.isfile(os.path.join(save_folder, f'{Data_type}.csv')):
        try:
            print('../database/' + Data_type + '.jsonl')
            noise_data = read_local_file(file_path='../database/' + Data_type + '.jsonl')
            if noise_type == 'test':
                conf = read_joblib('../database/config/' + 'bookid_' + noise_type + '_' + str(noise_ratio) + '.conf')
                suspected_seen_ids = conf['suspected_seen_id']
                suspected_unseen_ids = conf['suspected_unseen_id']
                print(suspected_seen_ids)
                print(suspected_unseen_ids)
                gt_label = []
                for item in noise_data:
                    if item['book_id'] in suspected_seen_ids:
                        gt_label.append(1)
                    elif item['book_id'] in suspected_unseen_ids:
                        gt_label.append(0)
                    else:
                        gt_label.append(0)
                print(len(gt_label))
            else:
                data = read_local_file(file_path='../database/my_bookmia.jsonl')
                data_dict = {(item['book_id'], item['snippet_id']): item['label'] for item in data}

                gt_label = [data_dict[(item['book_id'], item['snippet_id'])]
                            for item in noise_data
                            if (item['book_id'], item['snippet_id']) in data_dict]
            gt_label = np.array(gt_label)[0:(len(gt_label) // 4) * 4]
            noise_label = np.array([item['label'] for item in noise_data])[0:(len(noise_data) // 4) * 4]

            uc_metrics_dict = {'gt_label': gt_label,
                               'noise_label': noise_label}
            for epoch in range(EPOCH):
                columns_name = [item + str(epoch + 1) for item in
                                ['prob_', 'predictive_entropy_', 'predictive_kld_', 'predictive_std_', 'max_set_',
                                 'min_set_',
                                 'mean_med_set_', 'nll_set_', 'wd_set_', 'ed_set_', 'cd_set_', 'kdp_set_',
                                 'aleatoric_uc_',
                                 'epistemic_uc_']]
                data = collect_uc_metrics(model_type, sample_num, noise_label, Data_type, model_num, epoch=epoch + 1,
                                          llm_type=llm_type)

                aleatoric_uc_set = []
                epistemic_uc_set = []
                softmax_prob_set = np.transpose(np.array(data[-1]), (1, 0, 2))
                for i, item in enumerate(softmax_prob_set):
                    aleatoric_uncertainty_, epistemic_uncertainty_ = compute_variance(item)
                    aleatoric_uc_set.append(aleatoric_uncertainty_)
                    epistemic_uc_set.append(epistemic_uncertainty_)
                aleatoric_uc_set = np.array([item[0][0] for item in aleatoric_uc_set])
                epistemic_uc_set = np.array([item[1][1] for item in epistemic_uc_set])
                uc_data = data[:12] + [aleatoric_uc_set, epistemic_uc_set]
                print(np.array(uc_data).shape)
                for j, item in enumerate(columns_name):
                    uc_metrics_dict[item] = uc_data[j]
            pd.DataFrame(uc_metrics_dict).to_csv(os.path.join(save_folder, f'{Data_type}.csv'))
        except OSError as e:
            print('file not found')
            print(e)
    else:
        print(os.path.join(save_folder, f'{Data_type}.csv') + ' is exist!!')


if __name__ == '__main__':
    llm_type = 'open_llama_3b'
    model_type = 'deepensemble'
    sample_num = 10
    if model_type == 'blob':
        model_num = 1
    elif model_type == 'deepensemble':
        model_num = 10
    else:
        model_num = 1

    save_folder = os.path.join('feature_csv', model_type)
    save_folder = os.path.join(save_folder, llm_type)
    build_save_path()

    for noise_type in ['bonlyseen']:
        for noise_ratio in [10, 20, 30, 40]:
            main_(noise_ratio, noise_type, EPOCH=1, llm_type=llm_type)
    #
    # for noise_type in ['base10']:
    #     for noise_ratio in [0,100]:
    #         main_(noise_ratio, noise_type, EPOCH=1, llm_type=llm_type)

    # for noise_type in ['test']:
    #     for noise_ratio in ['seen10', 'seen20', 'seen30', 'seen40']:
    #         main_(noise_ratio, noise_type, EPOCH=1, llm_type=llm_type)
