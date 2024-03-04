from import_and_load import * 
from CV import get_groups, run_CV
from pca_analysis import * 
from train_test_split import *
from get_mean_variance import get_mean_variance


data, labels, groups = load_data('Features_Matrix.csv', 'Labels_Matrix.csv')
print(data.shape)
print(groups.shape)
print(labels.shape)
get_groups(data, labels, groups, 5)

train_data, test_data, train_labels, test_labels, train_groups, test_groups = train_test_split(data, labels, groups)
get_groups(train_data, train_labels, train_groups, 4)
KM_metrics_list, KM_f2_list, SVM_metrics_list, SVM_f2_list, RF_metrics_list, RF_f2_list = run_CV(data, labels, groups)

KM_f2_mean, KM_f2_var = get_mean_variance(KM_f2_list)
SVM_f2_mean, SVM_f2_var = get_mean_variance(SVM_f2_list)
RF_f2_mean, RF_f2_var = get_mean_variance(RF_f2_list)

print("K-Means F2 mean : ", KM_f2_mean)
print("K-Means F2 variance : ", KM_f2_var)
print("SVM F2 mean : ", SVM_f2_mean)
print("SVM F2 variance : ", SVM_f2_var)
print("Random Forest F2 mean : ", RF_f2_mean)
print("Random Forest F2 variance : ", RF_f2_var)