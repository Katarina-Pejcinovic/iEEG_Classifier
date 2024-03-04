
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from scaling_features import *
from pca_analysis import *
from sklearn.preprocessing import power_transform
from kmeans_clustering import *
from SVM import run_svm
from random_forest import run_random_forest
from performance_metrics import *
from tqdm import tqdm



def run_CV(data, labels, groups):

    stratified_group_kfold = StratifiedGroupKFold(n_splits=5, random_state=2, shuffle = True)
    counter = 0
    KM_f2_list = [None] * 5
    SVM_f2_list = [None] * 5
    RF_f2_list = [None] * 5
    KM_metrics_list = [None] * 5
    SVM_metrics_list = [None] * 5
    RF_metrics_list = [None] * 5

    for counter, (train_index, test_index) in enumerate(tqdm(stratified_group_kfold.split(data, labels, groups), total=5, desc="Processing")):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        scaled_X_train = scale_features(X_train)
        scaled_X_test = scale_features(X_test)

        np.random.seed(0)

        pca_X_train, pca_X_test = run_PCA(scaled_X_train, scaled_X_test)

        gaussian_pca_X_train = power_transform(pca_X_train)
        gaussian_pca_X_test = power_transform(pca_X_test)

        KM_predictions = run_KMeans(gaussian_pca_X_train, gaussian_pca_X_test)
        KM_metrics_list[counter] = get_performance_metrics(y_test, KM_predictions)
        KM_f2_list[counter] = KM_metrics_list[counter][-1]

        SVM_predictions = run_svm(gaussian_pca_X_train, gaussian_pca_X_test, y_train)
        SVM_metrics_list[counter] = get_performance_metrics(y_test, SVM_predictions)
        SVM_f2_list[counter] = SVM_metrics_list[counter][-1]

        RF_predictions = run_random_forest(gaussian_pca_X_train, gaussian_pca_X_test, y_train)
        RF_metrics_list[counter] = get_performance_metrics(y_test, RF_predictions)
        RF_f2_list[counter] = RF_metrics_list[counter][-1]

        print("fold", counter)

    return KM_metrics_list, KM_f2_list, SVM_metrics_list, SVM_f2_list, RF_metrics_list, RF_f2_list 



def get_groups(data, labels, groups): 

    sgkf = StratifiedGroupKFold(n_splits=5, random_state=2, shuffle = True)

    for i, (train_index, test_index) in enumerate(sgkf.split(data, labels, groups)):
        print(f"Fold {i}:")
    
        # Extract unique group IDs for the training and testing sets
        unique_train_group_ids = np.unique(groups[train_index])
        unique_test_group_ids = np.unique(groups[test_index])
        
        print(f"  Train: index={train_index}")
        print(f"         unique groups={unique_train_group_ids}")
        
        print(f"  Test:  index={test_index}")
        print(f"         unique groups={unique_test_group_ids}")
        print("")  # Add an empty line between folds