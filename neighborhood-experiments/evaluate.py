# -*- coding: utf-8 -*- 
# @Time : 2025/10/12 16:30 
# @Author : DirtyBoy 
# @File : evaluate_opensource_llm.py
import json, os
import numpy as np
from scipy.special import softmax
import pyarrow.parquet as pq
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def _check_probablities(p, q=None):
    assert 0. <= np.all(p) <= 1.
    if q is not None:
        assert len(p) == len(q), \
            'Probabilies and ground truth must have the same number of elements.'


def max_max2(end_list):
    max2 = np.sort(end_list)[-2]
    max = np.max(end_list)
    return max - max2


def min2_min(end_list):
    min2 = np.sort(end_list)[1]
    min = np.min(end_list)
    return min2 - min


def max_min(end_list):
    min = np.min(end_list)
    max = np.max(end_list)
    return max - min


def mean_med(end_list):
    mean = np.mean(end_list)
    med = np.median(end_list)
    return med - mean


def predictive_entropy(p, base=2, eps=1e-10, number=10):
    """
    calculate entropy in element-wise
    :param p: probabilities
    :param base: default exp
    :return: average entropy value
    """
    p_arr = np.asarray(p)
    _check_probablities(p)
    enc = -(p_arr * np.log(p_arr + eps) + (1. - p_arr) * np.log(1. - p_arr + eps))
    if base is not None:
        enc = np.clip(enc / np.log(base), a_min=0., a_max=1000)
    enc_ = []
    for item in enc:
        enc_.append([np.sum(item) / number])
    return np.mean(enc_)


def predictive_kld(p, number=10, w=None, base=2, eps=1e-10):
    """
    The Kullback-Leibler (KL) divergence measures the difference between two probability distributions by quantifying the information lost
    when one distribution is approximated by another. When comparing a probability vector to its mean vector, the KL divergence assesses
    the information difference between the original probabilities and the uniform distribution of their mean.


    calculate Kullback-Leibler divergence in element-wise
    :param p: probabilities
    :param number: the number of likelihood values for each sample
    :param w: weights for probabilities
    :param base: default exp
    :return: average entropy value
    """
    if number <= 1:
        return np.zeros_like(p)

    p_arr = np.asarray(p).reshape((-1, number))
    _check_probablities(p)
    q_arr = np.tile(np.mean(p_arr, axis=-1, keepdims=True), [1, number])
    if w is None:
        w_arr = np.ones(shape=(number, 1), dtype=float) / number
    else:
        w_arr = np.asarray(w).reshape((number, 1))

    kld_elem = p_arr * np.log((p_arr + eps) / (q_arr + eps)) + (1. - p_arr) * np.log(
        (1. - p_arr + eps) / (1. - q_arr + eps))
    if base is not None:
        kld_elem = kld_elem / np.log(base)
    kld = np.matmul(kld_elem, w_arr)
    return kld[0][0]


def nll(p, label, eps=1e-10, base=2):
    """
    negative log likelihood (NLL)
    :param p: predictive labels
    :param eps: a small value prevents the overflow
    :param base: the base of log function
    :return: the mean of NLL
    """
    p = np.array(p)
    q = np.full(len(p), label)
    nll = -(q * np.log(p + eps) + (1. - q) * np.log(1. - p + eps))
    if base is not None:
        nll = np.clip(nll / np.log(base), a_min=0., a_max=1000)
    return np.mean(nll)


def prob_label_kld(p, label, number=10, w=None, base=2, eps=1e-10):
    if number <= 1:
        return np.zeros_like(p)

    p_arr = np.asarray(p).reshape((-1, number))
    _check_probablities(p)
    q_arr = np.full(number, label)
    if w is None:
        w_arr = np.ones(shape=(number, 1), dtype=float) / number
    else:
        w_arr = np.asarray(w).reshape((number, 1))

    kld_elem = p_arr * np.log((p_arr + eps) / (q_arr + eps)) + (1. - p_arr) * np.log(
        (1. - p_arr + eps) / (1. - q_arr + eps))
    if base is not None:
        kld_elem = kld_elem / np.log(base)
    kld = np.matmul(kld_elem, w_arr)

    return (kld / number)[0][0]


def Wasserstein_distance(p, label):
    from scipy.stats import wasserstein_distance
    p = np.array(p)
    q = np.full(len(p), label)
    emd = wasserstein_distance(p, q)

    return emd


def Euclidean_distance(p, label):
    p = np.array(p)
    q = np.full(len(p), label)

    v1 = np.array(p)
    v2 = np.array(q)

    distance = np.linalg.norm(v1 - v2)

    return distance


def Manhattan_distance(p, label):
    p = np.array(p)
    q = np.full(len(p), label)
    v1 = np.array(p)
    v2 = np.array(q)
    distance = np.sum(np.abs(v1 - v2)) / len(p)

    return distance


def Chebyshev_distance(p, label):
    p = np.array(p)
    q = np.full(len(p), label)
    v1 = np.array(p)
    v2 = np.array(q)
    distance = np.max(np.abs(v1 - v2))

    return distance


def predictive_std(p, number=10, w=None):
    """
    calculate the probabilities deviation
    :param p: probabilities
    :param number: the number of probabilities applied to each sample
    :param w: weights for probabilities
    :param axis: the axis along which the calculation is conducted
    :return:
    """
    if number <= 1:
        return np.zeros_like(p)

    ps_arr = np.asarray(p).reshape((-1, number))
    _check_probablities(ps_arr)
    if w is None:
        w = np.ones(shape=(number, 1), dtype=float) / number
    else:
        w = np.asarray(w).reshape((number, 1))
    assert 0 <= np.all(w) <= 1.
    mean = np.matmul(ps_arr, w)
    var = np.sqrt(np.matmul(np.square(ps_arr - mean), w) * (float(number) / float(number - 1)))
    return var[0][0]


def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def new_label(outliers, X_feature, detection_algorithm_type):
    if detection_algorithm_type == 'gmm' or detection_algorithm_type == 'kmeans':
        a_l = 1
        b_l = 0
    elif detection_algorithm_type == 'if' or detection_algorithm_type == 'svm':
        a_l = 1
        b_l = -1
    else:
        a_l = -1
        b_l = 0

    a_index = np.where(outliers == a_l)[0]
    b_index = np.where(outliers == b_l)[0]
    a_uc = np.mean(X_feature[:, 0][a_index])
    b_uc = np.mean(X_feature[:, 0][b_index])
    if b_uc > a_uc:
        b_label = 0
        a_label = 1
    else:
        b_label = 1
        a_label = 0
    new_outliers = np.zeros(X_feature.shape[0])
    new_outliers[a_index] = a_label
    new_outliers[b_index] = b_label
    return new_outliers


def compute_variance(pt_samples):
    pt_mean = np.mean(pt_samples, axis=0)
    diag_terms = np.array([np.diag(pt) - np.outer(pt, pt) for pt in pt_samples])
    aleatoric_uncertainty = np.mean(diag_terms, axis=0)
    centered_terms = np.array([np.outer(pt - pt_mean, pt - pt_mean) for pt in pt_samples])
    epistemic_uncertainty = np.mean(centered_terms, axis=0)
    return aleatoric_uncertainty, epistemic_uncertainty


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


file_path = 'WikiMIA_24/WikiMIA_length64.parquet'
save_folder = os.path.join('logprobs', os.path.splitext(os.path.basename(file_path))[0])
if 'parquet' in file_path:
    table = pq.read_table(file_path).to_pandas().to_dict('records')
    key = 'input'
else:
    table = read_from_jsonl(file_path)
    if 'bookmia' in file_path:
        key = 'snippet'
    else:
        key = 'text'
gt_labels = np.array([item['label'] for item in table])
member_index = np.where(gt_labels == 1)[0]
nonmember_index = np.where(gt_labels == 0)[0]
print(len(nonmember_index))

my_data = []
for i in range(len(gt_labels)):
    data = read_json(os.path.join(save_folder, f'{i}.json'))

    tmp_probs = []
    for item in data:
        tmp_prob = softmax(np.array([item['0'], item['1']]))
        tmp_probs.append(tmp_prob)
    my_data.append(tmp_probs)
my_data = np.array(my_data)
uc = []
for item in my_data:
    tmp = []
    a, e = compute_variance(item)
    tmp.append(e[0][0])
    tmp.append(a[0][0])
    prob = [demo[1] for demo in item]
    tmp.append(predictive_entropy(prob))
    tmp.append(predictive_std(prob))
    tmp.append(predictive_kld(prob))
    tmp.append(max_max2(prob))
    tmp.append(mean_med(prob))
    tmp.append(min2_min(prob))
    tmp.append(max_min(prob))
    tmp.append(Chebyshev_distance(prob, label=1))
    tmp.append(Manhattan_distance(prob, label=1))
    tmp.append(Euclidean_distance(prob, label=1))
    tmp.append(Wasserstein_distance(prob, label=1))
    tmp.append(prob_label_kld(prob, label=1))
    tmp.append(nll(prob, label=1))
    uc.append(tmp)
X_feature = np.array(uc)
# uc = uc[:, 1]
# member_uc = uc[member_index]
# nonmember_uc = uc[nonmember_index]

# import matplotlib.pyplot as plt
#
# data = [member_uc, nonmember_uc]
# plt.boxplot(data, labels=["seen", "unseen"], showfliers=False)
# plt.ylabel("uncertainty")
# plt.title("")
# plt.show()
detection_algorithm_type = 'gmm'
scaler = StandardScaler()
X_feature = scaler.fit_transform(X_feature)

if detection_algorithm_type == 'if':
    isolation_forest = IsolationForest(n_estimators=50, contamination=0.2)
    isolation_forest.fit(X_feature)
    outliers = isolation_forest.predict(X_feature)
    print(list(outliers))

elif detection_algorithm_type == 'kmeans':
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_feature)
    outliers = kmeans.labels_
    print(list(outliers))

elif detection_algorithm_type == 'gmm':
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X_feature)
    outliers = gmm.predict(X_feature)
    print(list(outliers))

elif detection_algorithm_type == 'svm':
    from sklearn.svm import OneClassSVM

    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.5)
    ocsvm.fit(X_feature)

    outliers = ocsvm.predict(X_feature)
    print(list(outliers))
elif detection_algorithm_type == 'hierarchical':
    from sklearn.cluster import AgglomerativeClustering

    hier_cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
    outliers = hier_cluster.fit_predict(X_feature)

    print(list(outliers))
elif detection_algorithm_type == 'dbscan':
    dbscan = DBSCAN()
    dbscan.fit(X_feature)
    outliers = dbscan.labels_
    print(list(outliers))

else:
    outliers = None
outliers = new_label(outliers, X_feature, detection_algorithm_type)
print(len(gt_labels))
print(gt_labels)
print(list(outliers))
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

accuracy = accuracy_score(gt_labels, outliers)
MSG = "The accuracy on the dataset is {:.5f}%"
print(MSG.format(accuracy * 100))
balanced_accuracy = balanced_accuracy_score(gt_labels, outliers)
MSG = "The balanced accuracy on the dataset is {:.5f}%"
print(MSG.format(balanced_accuracy * 100))
tn, fp, fn, tp = confusion_matrix(gt_labels, outliers).ravel()
print(f"True Positives (tp): {tp} (Books correctly identified as 'seen') / {fn + tp}")
print(f"False Positives (fp): {fp} (Books incorrectly identified as 'unseen' but were 'seen') / {tn + fp}")
print(f"False Negatives (fn): {fn} (Books incorrectly identified as 'seen' but were 'unseen') / {fn + tp}")
print(f"True Negatives (tn): {tn} (Books correctly identified as 'unseen') / {tn + fp}")
