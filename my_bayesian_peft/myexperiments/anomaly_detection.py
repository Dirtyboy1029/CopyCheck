# -*- coding: utf-8 -*- 
# @Time : 2024/11/25 21:56 
# @Author : DirtyBoy 
# @File : anomaly_detection.py
import argparse, os
import numpy as np
from utils import evaluate, data_preprocessing, build_save_path, save_model, save_label_error_mask, \
    evaluate_label_noise_detection, pca
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

save_folder = 'label_error_masks/anomaly_detection'
save_model_path = 'models/anomaly_detection'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-noise_type', '-nt', type=str, default='bonlyseen')
    parser.add_argument('-detection_algorithm_type', '-dat', type=str, default='if')
    parser.add_argument('-noise_ratio', '-nr', type=str, default='40')
    parser.add_argument('-llm_type', '-lm', type=str, default='llama-7b')
    args = parser.parse_args()
    llm_type = args.llm_type
    detection_algorithm_type = args.detection_algorithm_type
    noise_type = args.noise_type
    noise_ratio = args.noise_ratio
    build_save_path(os.path.join(save_folder, llm_type))
    X_feature, gt_labels, noise_labels = data_preprocessing(noise_type, noise_ratio, llm_type)
    scaler = StandardScaler()
    X_feature = scaler.fit_transform(X_feature)
    if  os.path.isfile(os.path.join(os.path.join(save_folder, llm_type),
                                       f'{detection_algorithm_type}_{noise_type}_{noise_ratio}.npy')):
        if detection_algorithm_type == 'dbscan':
            dbscan = DBSCAN(eps=0.9, min_samples=1000)
            X_feature = pca(X_feature)
            dbscan.fit(X_feature)
            # save_model(dbscan, os.path.join(save_model_path, f'{detection_algorithm_type}_{noise_type}_{noise_ratio}.pkl'))
            outliers_dbscan = dbscan.labels_
            print(outliers_dbscan)
            print(len(outliers_dbscan))
            print(sum(outliers_dbscan))
            save_label_error_mask(outliers_dbscan,
                                  os.path.join(os.path.join(save_folder, llm_type),
                                               f'{detection_algorithm_type}_{noise_type}_{noise_ratio}'))
        elif detection_algorithm_type == 'if':
            isolation_forest = IsolationForest(n_estimators=100, contamination=0.5)
            X_feature = pca(X_feature)
            isolation_forest.fit(X_feature)
            outliers_isolation_forest = isolation_forest.predict(X_feature)
            print(list(outliers_isolation_forest))
            print(outliers_isolation_forest)
            print(len(outliers_isolation_forest))
            print(len(np.where(outliers_isolation_forest==1)[0]))
            outliers_isolation_forest = [0 if x == 1 else x for x in outliers_isolation_forest]

            save_label_error_mask(outliers_isolation_forest,
                                  os.path.join(os.path.join(save_folder, llm_type),
                                               f'{detection_algorithm_type}_{noise_type}_{noise_ratio}'))
        elif detection_algorithm_type == 'kmeans':
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=2, random_state=2)
            X_feature = pca(X_feature)
            kmeans.fit(X_feature)
            outliers_kmeans = kmeans.labels_
            print(list(outliers_kmeans))
            outliers_kmeans = [-1 if x == 1 else x for x in outliers_kmeans]
            # outliers_kmeans = [-1 if x == 0 else x for x in outliers_kmeans]
            save_label_error_mask(outliers_kmeans,
                                  os.path.join(os.path.join(save_folder, llm_type),
                                               f'{detection_algorithm_type}_{noise_type}_{noise_ratio}'))

        elif detection_algorithm_type == 'gmm':
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(n_components=2, random_state=42)
            X_feature = pca(X_feature)
            gmm.fit(X_feature)
            save_model(gmm,
                       os.path.join(save_model_path, f'{detection_algorithm_type}_{noise_type}_{noise_ratio}.pkl'))
            outliers_gmm = gmm.predict(X_feature)
            print(list(outliers_gmm))
            outliers_gmm = [-1 if x == 1 else x for x in outliers_gmm]
            save_label_error_mask(outliers_gmm,
                                  os.path.join(os.path.join(save_folder, llm_type),
                                               f'{detection_algorithm_type}_{noise_type}_{noise_ratio}'))
        label_error_mask = np.load(
            os.path.join(os.path.join(save_folder, llm_type),
                         f'{detection_algorithm_type}_{noise_type}_{noise_ratio}.npy'))
        evaluate_label_noise_detection(gt_labels, noise_labels, label_error_mask)
    else:
        label_error_mask = np.load(
            os.path.join(os.path.join(save_folder, llm_type),
                         f'{detection_algorithm_type}_{noise_type}_{noise_ratio}.npy'))
        evaluate_label_noise_detection(gt_labels, noise_labels, label_error_mask)
