# -*- coding: utf-8 -*- 
# @Time : 2024/11/12 21:12 
# 
# @File : utils.py
import json, os, ot
from scipy.special import softmax
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from metrics_utils import predictive_std, entropy, predictive_kld, nll, Wasserstein_distance, Euclidean_distance, \
    prob_label_kld, Chebyshev_distance, max_max2, min2_min, mean_med

from sklearn.neighbors import KernelDensity


def sliced_wasserstein_distance(S1, S2, w1=None, w2=None):
    if w1 is None:
        w1 = np.ones(len(S1)) / len(S1)
    if w2 is None:
        w2 = np.ones(len(S2)) / len(S2)

    n_seed = 20
    n_projections_arr = np.logspace(0, 3, 10, dtype=int)
    res = np.empty((n_seed, 10))
    for seed in range(n_seed):
        for i, n_projections in enumerate(n_projections_arr):
            res[seed, i] = ot.sliced_wasserstein_distance(
                S1, S2, w1, w2, n_projections, seed=seed
            )

    res_mean = np.mean(np.mean(res, axis=0))
    return res_mean


def caluate_wd(P, Q):
    kde_P = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(P)
    kde_Q = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Q)

    sampled_P = kde_P.sample(5000)
    sampled_Q = kde_Q.sample(5000)

    # 计算 Wasserstein 距离（通过样本点的距离）
    distance = wasserstein_distance(sampled_P.flatten(), sampled_Q.flatten())
    return distance


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


def collect_uc_metrics(model_type, sample_num, noise_labels, data_type, model_num, epoch, llm_type):
    prob_set = []
    softmax_prob_set = []
    for i in range(sample_num):
        model_prob, softmax_prob = get_blob_model_probs(
            '../output/' + model_type + '/' + data_type + f'_epoch{epoch}/{llm_type}-' + str(
                model_num) + '_' + str(i + 1) + '.data')
        prob_set.append(model_prob)
        softmax_prob_set.append(softmax_prob)
    tmp_prob_set = np.array(prob_set).T
    prob = np.array([np.mean(item) for item in tmp_prob_set])
    entropy_set = np.array([entropy(item, number=sample_num) for item in tmp_prob_set])
    kld_set = np.array([predictive_kld(item, number=sample_num) for item in tmp_prob_set])
    std_set = np.array([predictive_std(item, number=sample_num) for item in tmp_prob_set])
    nll_set = np.array([nll(item, label=noise_labels[i]) for i, item in enumerate(tmp_prob_set)])
    wd_set = np.array([Wasserstein_distance(item, label=noise_labels[i]) for i, item in enumerate(tmp_prob_set)])
    ed_set = np.array([Euclidean_distance(item, label=noise_labels[i]) for i, item in enumerate(tmp_prob_set)])
    cd_set = np.array([Chebyshev_distance(item, label=noise_labels[i]) for i, item in enumerate(tmp_prob_set)])
    kdp_set = np.array([prob_label_kld(item, label=noise_labels[i]) for i, item in enumerate(tmp_prob_set)])
    max_set = np.array([max_max2(item) for item in tmp_prob_set])
    min_set = np.array([min2_min(item) for item in tmp_prob_set])
    mean_med_set = np.array([mean_med(item) for item in tmp_prob_set])
    return [prob, entropy_set, kld_set, std_set, max_set, min_set, mean_med_set, nll_set, wd_set, ed_set, cd_set,
            kdp_set, softmax_prob_set]


def get_blob_model_probs(path):
    data = read_joblib(path)
    softmax_prob = [softmax(item) for item in data]
    model_prob = [item[1] for item in softmax_prob]
    return model_prob, softmax_prob


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


def evaluate(x_pred, gt_labels):
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
    accuracy = accuracy_score(gt_labels, x_pred)

    MSG = "The accuracy on the dataset is {:.5f}%"
    print(MSG.format(accuracy * 100))

    is_single_class = False
    if np.all(gt_labels == 1.) or np.all(gt_labels == 0.):
        is_single_class = True
    if not is_single_class:
        tn, fp, fn, tp = confusion_matrix(gt_labels, x_pred).ravel()
        fpr = fp / float(tn + fp)
        fnr = fn / float(tp + fn)
        f1 = f1_score(gt_labels, x_pred, average='binary')

        Recall = tp / (tp + fn)
        Specificity = tn / (tn + fp)
        IN = Recall + Specificity - 1
        precision = tp / (tp + fp)
        NPV = tn / (tn + fn)
        MK = precision + NPV - 1
        print("Other evaluation metrics we may need:")
        MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
        print(MSG.format(fnr * 100, fpr * 100, f1 * 100))
        MSG = "Recall is {:.5f}%, Prescion is {:.5f}%"
        print(MSG.format(Recall * 100, precision * 100))


def calculate_wasserstein_distance(S1, S2, w1=None, w2=None):
    if w1 is None:
        w1 = np.ones(len(S1)) / len(S1)
    if w2 is None:
        w2 = np.ones(len(S2)) / len(S2)

    M = ot.dist(S1, S2)
    G = ot.sinkhorn(w1, w2, M, reg=0.1)
    wasserstein_distance = np.sum(G * M)
    return wasserstein_distance


def save_model(model, model_path):
    import joblib
    joblib.dump(model, model_path)


def save_label_error_mask(label_error_mask, mask_path):
    np.save(mask_path, label_error_mask)


def build_save_path(save_folder):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)


def main_(model_type, noise_type, noise_ratio, llm_type):
    tmp_list = ['prob_1', 'predictive_entropy_1', 'predictive_kld_1', 'predictive_std_1', 'max_set_1', 'min_set_1',
                'mean_med_set_1', 'nll_set_1', 'wd_set_1', 'ed_set_1', 'cd_set_1', 'kdp_set_1',
                'aleatoric_uc_1', 'epistemic_uc_1']
    data1 = pd.read_csv(f'feature_csv/{model_type}/{llm_type}/bookmia_{noise_type}_{noise_ratio}.csv')
    del data1['Unnamed: 0']
    gt_labels = np.array(data1['gt_label'].tolist())
    noise_labels = np.array(data1['noise_label'].tolist())
    suspect_index = np.where(noise_labels == 1)[0]
    data1 = data1.loc[:, tmp_list]
    data1 = data1.fillna(0).values
    suspect_samples1 = data1[suspect_index]
    return suspect_samples1


def data_preprocessing(noise_type, noise_ratio, llm_type, is_anomaly=True, do_pca=True):
    if is_anomaly:
        data1 = pd.read_csv(f'feature_csv/blob/{llm_type}/bookmia_{noise_type}_{noise_ratio}.csv')
        del data1['Unnamed: 0']
        gt_labels = np.array(data1['gt_label'].tolist())
        noise_labels = np.array(data1['noise_label'].tolist())
        suspect_index = np.where(noise_labels == 1)[0]
        data_list = []
        for model_type in ['blob', 'mcdropout', 'deepensemble']:  #
            data_list.append(main_(model_type, noise_type, noise_ratio, llm_type))
        suspect_samples = np.hstack(data_list)
        suspect_gt_labels = gt_labels[suspect_index]
        suspect_noise_labels = noise_labels[suspect_index]
        return suspect_samples, suspect_gt_labels, suspect_noise_labels
    else:
        columns_list = ['prob_1', 'predictive_entropy_1', 'predictive_kld_1', 'predictive_std_1', 'max_set_1',
                        'min_set_1', 'nll_set_1', 'wd_set_1', 'ed_set_1', 'cd_set_1', 'kdp_set_1',
                        'aleatoric_uc_1', 'epistemic_uc_1']
        unseen_dfs = []
        base_seen_dfs = []
        test_dfs = []
        for model_type in ['blob', 'mcdropout', 'deepensemble']:
            unseen_df, base_seen_df, test_df = split_feature(model_type=model_type, noise_ratio=noise_ratio,
                                                             noise_type=noise_type, do_pca=do_pca, llm_type=llm_type)
            if not do_pca:
                if model_type == 'blob':
                    unseen_df = unseen_df.loc[:, ['gt_label', 'noise_label'] + columns_list]
                    base_seen_df = base_seen_df.loc[:, ['gt_label', 'noise_label'] + columns_list]
                    test_df = test_df.loc[:, ['gt_label', 'noise_label'] + columns_list]
                else:
                    unseen_df = unseen_df.loc[:, columns_list]
                    base_seen_df = base_seen_df.loc[:, columns_list]
                    test_df = test_df.loc[:, columns_list]
            else:
                if model_type != 'blob':
                    unseen_df = unseen_df.iloc[:, 2:]
                    base_seen_df = base_seen_df.iloc[:, 2:]
                    test_df = test_df.iloc[:, 2:]
            unseen_df = unseen_df.add_prefix(f"{model_type}_")
            base_seen_df = base_seen_df.add_prefix(f"{model_type}_")
            test_df = test_df.add_prefix(f"{model_type}_")

            unseen_dfs.append(unseen_df)
            base_seen_dfs.append(base_seen_df)
            test_dfs.append(test_df)
        unseen_df = pd.concat(unseen_dfs, axis=1)
        base_seen_df = pd.concat(base_seen_dfs, axis=1)
        test_df = pd.concat(test_dfs, axis=1)
        return unseen_df, base_seen_df, test_df


def evaluate_label_noise_detection(true_labels, noisy_labels, label_error_mask):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    predicted_noise = (noisy_labels != true_labels).astype(int)
    print("Accuracy of source label noise: {:.5f}%".format(accuracy_score(true_labels, noisy_labels) * 100))
    label_error_mask = (label_error_mask == -1).astype(int)
    accuracy = accuracy_score(label_error_mask, predicted_noise)
    print("Accuracy of label noise detection: {:.5f}%".format(accuracy * 100))
    tn, fp, fn, tp = confusion_matrix(label_error_mask, predicted_noise).ravel()
    fpr = fp / float(tn + fp)
    fnr = fn / float(tp + fn)
    precision = precision_score(label_error_mask, predicted_noise)
    recall = recall_score(label_error_mask, predicted_noise)
    f1 = f1_score(label_error_mask, predicted_noise)

    print("Other evaluation metrics:")
    print(f"False Positive Rate (FPR): {fpr * 100:.5f}%")
    print(f"False Negative Rate (FNR): {fnr * 100:.5f}%")
    print(f"Precision: {precision * 100:.5f}%")
    print(f"Recall: {recall * 100:.5f}%")
    print(f"F1 score: {f1 * 100:.5f}%")


def pca(data):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    pca = PCA(n_components=8)
    reduced_data = pca.fit_transform(data_standardized)
    print("方差解释率：", pca.explained_variance_ratio_)

    # 累积方差解释率
    print("累积方差解释率：", np.cumsum(pca.explained_variance_ratio_))
    return reduced_data


def data2point(df, epoch):
    x_name = 'aleatoric_uc_' + str(epoch)
    y_name = 'epistemic_uc_' + str(epoch)
    return np.array([df[x_name], df[y_name]]).T


def get_samples_index(source_data, book_ids):
    tmp_id = []
    for id in book_ids:
        for i, item in enumerate(source_data):
            if item['book_id'] == id:
                tmp_id.append(i)
    return tmp_id


def split_feature(model_type, noise_ratio, noise_type, llm_type, do_pca=True):
    source_data = read_local_file(f'../database/bookmia_{noise_type}_{noise_ratio}.jsonl')
    data = pd.read_csv(f'feature_csv/{model_type}/{llm_type}/bookmia_{noise_type}_{noise_ratio}.csv')
    del data['Unnamed: 0']
    if not do_pca:
        source_data = source_data[:data.shape[0]]
        data = data.fillna(0)
    else:
        source_data = source_data[:data.shape[0]]
        data = data.fillna(0)
        X_data = data.iloc[:, 2:].values

        X_data = pca(X_data)
        print(X_data.shape)
        data1 = pd.DataFrame({'gt_label': data['gt_label'].tolist(),
                              'noise_label': data['noise_label'].tolist()})
        tmp_source_data = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(10)])
        data = pd.concat([data1.reset_index(drop=True), tmp_source_data.reset_index(drop=True)], axis=1)
        print(data.shape)
    noise_type = noise_type.replace('both', '')
    conf = read_joblib(f'../database/config/bookid_{noise_type}_{noise_ratio}.conf')
    unseen_bookid = conf['unseen_id']
    base_seen_bookid = conf['base_seen_id']
    noise_seen_bookid = conf['noise_seen_id']
    seen_bookid = conf['seen_id']

    unseen_samples_indexs = get_samples_index(source_data, unseen_bookid)
    noise_seen_samples_indexs = get_samples_index(source_data, noise_seen_bookid)
    base_seen_samples_indexs = get_samples_index(source_data, base_seen_bookid)
    seen_samples_indexs = get_samples_index(source_data, seen_bookid)

    return data.iloc[unseen_samples_indexs], data.iloc[base_seen_samples_indexs], data.iloc[
        seen_samples_indexs + noise_seen_samples_indexs]


def data_processing_df(unseen_df):
    X = unseen_df.iloc[:, 2:].values

    y = np.array(unseen_df['blob_gt_label'].tolist())
    return X, y
