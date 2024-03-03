
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from performance_metrics import *
from SVM import run_svm

def run_CV(data, labels, groups, model):

    stratified_group_kfold = StratifiedGroupKFold(n_splits=2, random_state=2, shuffle = True)
    counter = 0
    for train_index, test_index in stratified_group_kfold.split(data, labels, groups):
        counter +=1 
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        if model == 'SVM':
            predictions = run_svm(X_train, y_train)

        metrics = get_performance_metrics(y_train, predictions)

        print(metrics)
    return metrics 

# Example usage:
# Assuming 'your_data', 'your_labels', and 'your_groups' are your input data, labels, and groups
# Replace 'your_data', 'your_labels', 'your_groups' with your actual data, labels, and groups
# Replace 'your_model' with the model you want to evaluate
your_data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [4, 5], [6, 7]])
your_labels = np.array([0, 1, 0, 1, 0 , 0])
your_groups = np.array([1, 2, 3, 4, 1, 2])

# The result will be a list of accuracy scores for each fold
accuracy_scores = run_CV(your_data, your_labels, your_groups, 'SVM')

print("Accuracy Scores:", accuracy_scores)

def get_groups(data, labels, groups): 
    sgkf = StratifiedGroupKFold(n_splits=2, random_state=2, shuffle = True)
    
    for i, (train_index, test_index) in enumerate(sgkf.split(data, labels, groups)):
         print(f"Fold {i}:")
         print(f"  Train: index={train_index}")
         print(f"         group={groups[train_index]}")
         print(f"  Test:  index={test_index}")
         print(f"         group={groups[test_index]}")

get_groups(your_data, your_labels, your_groups)