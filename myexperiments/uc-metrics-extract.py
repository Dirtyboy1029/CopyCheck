#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: DirtyBoy
@Date: 2026/7/25 10:18
"""
import os, joblib, json
from metrics_utils import *
from scipy.special import softmax
import pandas as pd

import numpy as np


def array_to_ucmetrics(my_array: np.array, suspect_labels: np.array):
    softmax_array = np.array(softmax(my_array, axis=-1))
    aleatoric_uc_set, epistemic_uc_set = map(np.array, zip(*(compute_variance(item) for item in softmax_array)))

    my_prob = softmax_array[:, :, 0]
    prob_set = np.array([np.mean(item) for item in my_prob])
    entropy_set = np.array([entropy(item, number=10) for item in my_prob])
    kld_set = np.array([predictive_kld(item, number=10) for item in my_prob])
    std_set = np.array([predictive_std(item, number=10) for item in my_prob])
    nll_set = np.array([nll(item, label=suspect_labels[i]) for i, item in enumerate(my_prob)])
    wd_set = np.array([Wasserstein_distance(item, label=suspect_labels[i]) for i, item in enumerate(my_prob)])
    ed_set = np.array([Euclidean_distance(item, label=suspect_labels[i]) for i, item in enumerate(my_prob)])
    cd_set = np.array([Chebyshev_distance(item, label=suspect_labels[i]) for i, item in enumerate(my_prob)])
    kdp_set = np.array([prob_label_kld(item, label=suspect_labels[i]) for i, item in enumerate(my_prob)])
    max_set = np.array([max_max2(item) for item in my_prob])
    min_set = np.array([min2_min(item) for item in my_prob])
    mean_med_set = np.array([mean_med(item) for item in my_prob])

    return {'prob': prob_set,
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


def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def read_joblib(path):
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def main_(data_type="book",
          ratio=10,
          model_name='huggyllama/llama-7b',
          L=300):
    data_name = f"bookmia_{ratio}_{data_type}"
    save_path = f"uc_metrics_csv_{L}/{model_name}/{data_name}.csv"
    if not os.path.isfile(save_path):
        source_data = read_from_jsonl(f"../Database/{data_name}.jsonl")

        suspect_snippet_labels = np.array([item['target'] for item in source_data])

        my_dict = []
        for i, uc_type in enumerate(['blob',"mcdropout",]):  #   'deepensemble'
            tmp_array = np.array(
                [read_joblib(
                    f'../bayesian_peft/output/causallm/{uc_type}/{data_name}/{data_name}/epoch1/{model_name}-{i + 1}.data')
                    for
                    i in range(10)]).transpose(1, 0, 2)
            tmp_dict = array_to_ucmetrics(tmp_array, suspect_snippet_labels)
            my_dict.append({f"{uc_type}_{k}": v for k, v in tmp_dict.items()})

        my_df = my_dict[0] | my_dict[1]  # | my_dict[2]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame(my_df).to_csv(save_path, index=False)
        print("uc metrics file save to ", save_path)
    else:
        print("The file {0} already exists.".format(save_path))


if __name__ == '__main__':

    L = 300

    for data_type in ["book", "arxiv1823", "arxiv823", "arxiv1825", "arxiv825"]:
        for model_name in ['EleutherAI/gpt-neox-20b']:
            # ['huggyllama/llama-7b', "meta-llama/Llama-2-7b", , "meta-llama/Llama-2-13b"
            #            "EleutherAI/gpt-j-6b", ]:  # 'facebook/opt-6.7b', "facebook/opt-2.7b"
            for ratio in [25]:
                try:
                    main_(data_type=data_type,
                          ratio=ratio,
                          model_name=model_name,
                          L=L)
                except Exception as e:
                    print(e)

    # L = 300
    #
    # for data_type in ["arxiv818", "arxiv1823", "arxiv823"]:
    #     for model_name in ["meta-llama/Llama-3.1-8B"]:
    #         for ratio in [25, ]:
    #             try:
    #                 main_(data_type=data_type,
    #                       ratio=ratio,
    #                       model_name=model_name,
    #                       L=L)
    #             except Exception as e:
    #                 print(e)

# cp -r /home/nusbac/LHD_LLM/CopyCheck/bayesian_peft/output/causallm/deepensemble/bookmia_25_arxiv* /home/nusbac/LHD_LLM/CopyCheck/bayesian_peft/output-300/causallm/deepensemble/
