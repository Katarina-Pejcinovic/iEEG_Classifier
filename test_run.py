from import_and_load import * 
from CV import get_groups, run_CV
from pca_analysis import * 
from train_test_split import *

data, labels, groups = load_data('Features_Matrix.csv', 'Labels_Matrix.csv')
print(data.shape)
print(groups.shape)
print(labels.shape)
get_groups(data, labels, groups, 5)

data_PCA = run_PCA(data, 39)
train_data, test_data, train_labels, test_labels, train_groups, test_groups = train_test_split(data_PCA, labels, groups)
get_groups(train_data, train_labels, train_groups, 2)
metrics = run_CV(data_PCA, labels, groups, 'SVM', 2)
