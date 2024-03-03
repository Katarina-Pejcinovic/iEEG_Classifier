
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from performance_metrics import *
from SVM import run_svm
from tqdm import tqdm



def run_CV(data, labels, groups, model):

    stratified_group_kfold = StratifiedGroupKFold(n_splits=5, random_state=2, shuffle = True)
    counter = 0
    
    for counter, (train_index, test_index) in enumerate(tqdm(stratified_group_kfold.split(data, labels, groups), total=5, desc="Processing")):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        if model == 'SVM':
            predictions = run_svm(X_train, X_test, y_train)
        print("fold", counter)
        metrics = get_performance_metrics(y_test, predictions)

        print(metrics)
    return metrics 



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