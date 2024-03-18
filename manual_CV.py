import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
import pandas as pd 
from pca_analysis import run_PCA
from scaling_features import *
from sklearn.preprocessing import power_transform
from SVM import *
from random_forest import * 
from kmeans_clustering import *
from performance_metrics import *
from sklearn.metrics import confusion_matrix
from confusion_matrix import make_confusion_matrix

def manual_cross_val(data_folds, label_folds):

    num_folds = len(data_folds)
    counter = 0
    KM_f2_list = [None] * 5
    SVM_f2_list = [None] * 5
    RF_f2_list = [None] * 5
    KM_metrics_list = [None] * 5
    SVM_metrics_list = [None] * 5
    RF_metrics_list = [None] * 5

    #initialize empty arrays to save predicted and true values over all folds 
    concat_true = np.array([])
    concat_pred_KM = np.array([])
    concat_pred_SVM = np.array([])
    concat_pred_RF = np.array([])
    # Perform 4-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(range(num_folds)):
        
        # Select training and testing data for this fold
        train_data, test_data = np.concatenate([data_folds[i] for i in train_index]), np.concatenate([data_folds[i] for i in test_index])
        train_labels, test_labels = np.concatenate([label_folds[i] for i in train_index]), np.concatenate([label_folds[i] for i in test_index])
        concat_true = np.concatenate((concat_true, test_labels), axis = 0)
        #print(f"size of true labels {counter})", concat_true.shape)
        print(train_data.shape)
        print(train_labels.shape)
        print(test_data.shape)
        print(test_labels.shape)

        scaled_X_train = scale_features(train_data)
        scaled_X_test = scale_features(test_data)

        np.random.seed(0)
        
        pca_X_train, pca_X_test = run_PCA(scaled_X_train, scaled_X_test)

        gaussian_pca_X_train = power_transform(pca_X_train)
        gaussian_pca_X_test = power_transform(pca_X_test)

        print("fold", counter)

        print("KM predictions")
        KM_predictions = run_KMeans(gaussian_pca_X_train, gaussian_pca_X_test, scaled_X_train, scaled_X_test)
        KM_metrics_list[counter] = get_performance_metrics(test_labels, KM_predictions)
        KM_f2_list[counter] = KM_metrics_list[counter][-1]
        concat_pred_KM = np.concatenate((concat_pred_KM, KM_predictions), axis = 0)


        print("SVM predictions")
        SVM_predictions = run_svm(gaussian_pca_X_train, gaussian_pca_X_test, train_labels)
        SVM_metrics_list[counter] = get_performance_metrics(test_labels, SVM_predictions)
        SVM_f2_list[counter] = SVM_metrics_list[counter][-1]
        concat_pred_SVM = np.concatenate((concat_pred_SVM, SVM_predictions), axis = 0)

        print("RF predictions")
        RF_predictions = run_random_forest(gaussian_pca_X_train, gaussian_pca_X_test, train_labels)
        RF_metrics_list[counter] = get_performance_metrics(test_labels, RF_predictions)
        RF_f2_list[counter] = RF_metrics_list[counter][-1]
        concat_pred_RF = np.concatenate((concat_pred_RF, RF_predictions), axis = 0)

        counter += 1

    conf_matrix_KM = confusion_matrix(concat_true, concat_pred_KM)
    conf_matrix_SVM = confusion_matrix(concat_true, concat_pred_SVM)
    conf_matrix_RF = confusion_matrix(concat_true, concat_pred_RF)

    np.savetxt('confusion_matricies_KM.txt', conf_matrix_KM, '%8d')
    np.savetxt('confusion_matricies_SVM.txt', conf_matrix_SVM, '%8d')
    np.savetxt('confusion_matricies_RF.txt', conf_matrix_RF, '%8d')
    
    labels = ['True Negative','False Positive','False Negative','True Positive']
    categories = ['Physiological', 'Pathological']
    make_confusion_matrix(conf_matrix_KM, 
                        group_names=labels,
                        categories=categories, 
                        title='K-Means Confusion Matrix',
                        sum_stats=False)
    make_confusion_matrix(conf_matrix_SVM, 
                        group_names=labels,
                        categories=categories, 
                        title='SVM Confusion Matrix',
                        sum_stats=False)
    make_confusion_matrix(conf_matrix_RF, 
                        group_names=labels,
                        categories=categories, 
                        title='Random Forest Confusion Matrix',
                        sum_stats=False)
    
    return KM_metrics_list, KM_f2_list, SVM_metrics_list, SVM_f2_list, RF_metrics_list, RF_f2_list 


def get_np_folds(data_folds, label_folds):
    all_folds_data = []
    for fold in range(5):
        print("FOLD", fold)
        #get first patient data. We will concatenate the other patients in the fold to this
        data = np.array(data_folds[fold][0])
        for patient in range(1, len(data_folds[fold])):
            print("patient", patient)
            data = np.concatenate((data, np.array(data_folds[fold][patient])), axis = 0)
    
        print("data shape after concat", data.shape)
        all_folds_data.append(data)

    print("doing labels")
    all_folds_labels = []
    for fold in range(5):
        print("FOLD", fold)
        #get first patient data. We will concatenate the other patients in the fold to this
        labels = np.array(label_folds[fold][0])
        for patient in range(1, len(label_folds[fold])):
            print("patient", patient)
            labels = np.concatenate((labels, np.array(label_folds[fold][patient])), axis = 0)
    
        print("data shape after concat", labels.shape)
        all_folds_labels.append(labels)
        print("--------------")

    return all_folds_data, all_folds_labels
    
