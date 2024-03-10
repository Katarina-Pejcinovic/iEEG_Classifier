import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score  # You can replace this with the appropriate metric for your problem
from sklearn.ensemble import RandomForestClassifier  # You can replace this with the model of your choice
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

def manual_cross_val(data_folds, label_folds):

    num_folds = len(data_folds)
    counter = 0
    KM_f2_list = [None] * 4
    SVM_f2_list = [None] * 4
    RF_f2_list = [None] * 4
    KM_metrics_list = [None] * 4
    SVM_metrics_list = [None] * 4
    RF_metrics_list = [None] * 4

    #initialize empty arrays to save predicted and true values over all folds 
    concat_true = np.array([])
    concat_pred_KM = np.array([])
    concat_pred_SVM = np.array([])
    concat_pred_RF = np.array([])
    # Perform 4-fold cross-validation
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
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
        KM_predictions = run_KMeans(gaussian_pca_X_train, gaussian_pca_X_test)
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
    
    return KM_metrics_list, KM_f2_list, SVM_metrics_list, SVM_f2_list, RF_metrics_list, RF_f2_list 


def get_np_folds(data_folds, label_folds):
    all_folds_data = []
    for fold in range(4):
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
    for fold in range(4):
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
    

'''#separate data into 5 groups'''

features_matrix = np.genfromtxt('/Users/andresmichel/Desktop/Features Matrix.csv', delimiter=',')
labels_matrix = np.genfromtxt('/Users/andresmichel/Desktop/Labels Matrix.csv', delimiter=',')


# Extract Pandas DF with only Physio and Patho Segments
segments_file = pd.read_csv('/Users/andresmichel/Desktop/DATASET_MAYO/segments.csv')

# print(segments_file)
segments_df = segments_file[segments_file['category_id'].isin([2,3])]
segments_df = segments_df.reset_index(drop=True)
segments_df = segments_df.drop(columns=['index'])

# create indices list for all 24 patients with the indices for physio and patho segments
physio_patho_labels = labels_matrix[:,0]
print(physio_patho_labels)
soz_labels = labels_matrix[:,1]

indices_list = [np.empty((0,))] * 24

for i in range(len(segments_df)):        
    p = 0
    while p < 24:
        #if segment is from patient p
        if segments_df['patient_id'][i] == p:
            #append index to matrix
            indices_list[p] = np.append(indices_list[p], [i], axis=0)
            break
        p+=1


# create features list with all 24 patients' features
features_list = [None] * 24

for i in range(24):
    features_list[i] = features_matrix[indices_list[i].astype(int)]

# create labels list with all 24 patients' labels
labels_list = [None] * 24

for i in (range(24)):
    labels_list[i] = physio_patho_labels[indices_list[i].astype(int)]


'''create folds'''
#fold 1 = 0, 1, 2 (testing set)
fold_1_data =[features_list[0], features_list[1], features_list[2]]
fold_1_labels = [labels_list[0], labels_list[1], labels_list[2]]

#fold 2 = 3, 17, 23
fold_2_data = [features_list[3], features_list[17], features_list[23]]
fold_2_labels = [labels_list[3], labels_list[17], labels_list[23]]

#fold 3 = 4,8 ,16, 18
fold_3_data = [features_list[4], features_list[8], features_list[16], features_list[18]]
fold_3_labels = [labels_list[4], labels_list[8], labels_list[16], labels_list[18]]

#fold 4 = 5, 14, 20 
fold_4_data = [features_list[5], features_list[14], features_list[20]]
fold_4_labels = [labels_list[5], labels_list[14], labels_list[20]]

#fold 5 = 7, 21
fold_5_data = [features_list[7], features_list[21]]
fold_5_labels = [labels_list[7], labels_list[21]]

data_folds = [fold_2_data, fold_3_data, fold_4_data, fold_5_data]
label_folds = [fold_2_labels, fold_3_labels, fold_4_labels, fold_5_labels]

print("made folds")
#KM_metrics, KM_f2, SVM_metrics, SVM_f2, RF_metrics, RF_f2 = manual_cross_val(data_folds, label_folds)
#print(KM_metrics, KM_f2, SVM_metrics, SVM_f2, RF_metrics, RF_f2)

data_folds, label_folds = get_np_folds(data_folds, label_folds)
print("len data", len(data_folds), len(label_folds))

KM_metrics_list, KM_f2_list, SVM_metrics_list, SVM_f2_list, RF_metrics_list, RF_f2_list  = manual_cross_val(data_folds, label_folds)
print("KM", '\n', KM_metrics_list, KM_f2_list, )
print("SVM", '\n', SVM_metrics_list, SVM_f2_list, )
print("RF", RF_metrics_list, RF_f2_list)