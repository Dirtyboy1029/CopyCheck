# -*- coding: utf-8 -*- 
# @Time : 2024/12/15 13:55 
# @Author : DirtyBoy 
# @File : classification_detection.py
import os, json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def get_samples_index(source_data, book_ids):
    my_len = len(source_data) - (len(source_data) % 4)

    tmp_id = []
    for id in book_ids:
        for i, item in enumerate(source_data):
            if item['book_id'] == id and i < my_len:
                tmp_id.append(i)
    return tmp_id


def read_local_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def data_preprocessing(noise_type, noise_ratio, llm_type):
    columns_list = ['predictive_entropy_1', 'predictive_kld_1', 'predictive_std_1', 'max_set_1',
                    'min_set_1', 'nll_set_1', 'wd_set_1', 'ed_set_1', 'cd_set_1', 'kdp_set_1',
                    'aleatoric_uc_1', 'epistemic_uc_1']
    train_dfs = []
    test_dfs = []
    for model_type in ['blob', 'mcdropout', 'deepensemble']:
        train_df, test_df = split_feature(model_type=model_type, noise_type=noise_type, noise_ratio=noise_ratio,
                                          llm_type=llm_type)

        if model_type == 'blob':
            train_df = train_df.loc[:, ['gt_label', 'noise_label'] + columns_list]
            test_df = test_df.loc[:, ['gt_label', 'noise_label'] + columns_list]
        else:
            train_df = train_df.loc[:, columns_list]
            test_df = test_df.loc[:, columns_list]
        train_df = train_df.add_prefix(f"{model_type}_")
        test_df = test_df.add_prefix(f"{model_type}_")
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    train_df = pd.concat(train_dfs, axis=1)
    test_df = pd.concat(test_dfs, axis=1)
    return train_df, test_df


def pca(train_df, test_df, n_components):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    train_features = train_df.iloc[:, 2:]
    test_features = test_df.iloc[:, 2:]

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_features_scaled)
    # print("方差解释率：", pca.explained_variance_ratio_)
    #
    # # 累积方差解释率
    # print("累积方差解释率：", np.cumsum(pca.explained_variance_ratio_))
    test_pca = pca.transform(test_features_scaled)

    train_df_pca = pd.concat([
        train_df.iloc[:, :2].reset_index(drop=True),
        pd.DataFrame(train_pca, columns=[f'PC{i + 1}' for i in range(n_components)])
    ], axis=1)

    test_df_pca = pd.concat([
        test_df.iloc[:, :2].reset_index(drop=True),
        pd.DataFrame(test_pca, columns=[f'PC{i + 1}' for i in range(n_components)])
    ], axis=1)

    return train_df_pca, test_df_pca


def split_feature(model_type, noise_type, noise_ratio, llm_type):
    source_data = read_local_file(f'../database/bookmia_{noise_type}_test{noise_ratio}.jsonl')
    data = pd.read_csv(f'feature_csv/{model_type}/{llm_type}/bookmia_{noise_type}_test{noise_ratio}.csv')
    del data['Unnamed: 0']
    data = data.fillna(0)
    conf = read_joblib(f'../database/config/bookid_{noise_type}_test{noise_ratio}.conf')

    base_seen_bookid = conf['base_seen_id']
    base_unseen_bookid = conf['base_unseen_id']
    test_seen_bookid = conf['test_seen_id']
    test_unseen_bookid = conf['test_unseen_id']

    base_seen_id = get_samples_index(source_data, base_seen_bookid)
    base_unseen_id = get_samples_index(source_data, base_unseen_bookid)
    test_seen_id = get_samples_index(source_data, test_seen_bookid)
    test_unseen_id = get_samples_index(source_data, test_unseen_bookid)

    return data.iloc[base_seen_id + base_unseen_id], data.iloc[test_seen_id + test_unseen_id]


def data_processing_df(unseen_df):
    X = unseen_df.iloc[:, 2:].values
    y = np.array(unseen_df['blob_gt_label'].tolist())
    return X, y


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


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def save_label_error_mask(label_error_mask, mask_path):
    np.save(mask_path, label_error_mask)


if __name__ == '__main__':
    noise_type = 'base10'
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-detection_algorithm_type', '-dat', type=str, default='svm')
    parser.add_argument('-noise_ratio', '-nr', type=int, default=55)  #### [10,32,55,77]
    parser.add_argument('-llm_type', '-lt', type=str, default='llama2-7b')  #### [10,32,55,77]
    args = parser.parse_args()
    llm_type = args.llm_type
    detection_algorithm_type = args.detection_algorithm_type
    noise_ratio = args.noise_ratio
    train_df, test_df = data_preprocessing(noise_type, noise_ratio, llm_type=llm_type)
    train_df, test_df = pca(train_df, test_df, n_components=10)
    gt_label = np.array(test_df['blob_gt_label'].tolist())
    noise_label = np.array(test_df['blob_noise_label'].tolist())
    mask = (noise_label != gt_label).astype(int)
    train_data, train_y = data_processing_df(train_df)
    test_data, test_y = data_processing_df(test_df)
    save_folder = 'label_error_masks/binary_classification'
    if not os.path.isfile(os.path.join(os.path.join(save_folder, llm_type),
                                       f'{detection_algorithm_type}_{noise_type}_{noise_ratio}' + '.npy')):
        if detection_algorithm_type == 'svm':
            from sklearn.svm import SVC

            svm_model = SVC(kernel='linear', C=1000, random_state=42)
            svm_model.fit(train_data, train_y)
            y_pred = svm_model.predict(test_data)
        elif detection_algorithm_type == 'lr':
            from sklearn.linear_model import LogisticRegression

            lr_model = LogisticRegression(C=0.00001, random_state=42)
            lr_model.fit(train_data, train_y)
            y_pred = lr_model.predict(test_data)
        elif detection_algorithm_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier

            knn_model = KNeighborsClassifier(n_neighbors=3)
            knn_model.fit(train_data, train_y)
            y_pred = knn_model.predict(test_data)
        elif detection_algorithm_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(train_data, train_y)
            y_pred = rf_model.predict(test_data)
        else:
            y_pred = np.zeros(len(test_y))
            raise Exception

        save_label_error_mask(1 - y_pred,
                              os.path.join(os.path.join(save_folder, llm_type),
                                           f'{detection_algorithm_type}_{noise_type}_{noise_ratio}'))
        print(f'Source Acc:{accuracy_score(test_y, noise_label) * 100:.5f}%', )
        evaluate(1 - y_pred, mask)
    else:
        print('load file from ' + os.path.join(os.path.join(save_folder, llm_type),
                                               f'{detection_algorithm_type}_{noise_type}_{noise_ratio}' + '.npy'))
        y_pred = np.load(os.path.join(os.path.join(save_folder, llm_type),
                                      f'{detection_algorithm_type}_{noise_type}_{noise_ratio}' + '.npy'))
        evaluate(y_pred, mask)
