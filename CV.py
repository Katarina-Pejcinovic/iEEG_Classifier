
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from performance_metrics import *
from SVM import run_svm
from tqdm import tqdm
from random_forest import run_random_forest



def run_CV(data, labels, groups, model, splits):

    stratified_group_kfold = StratifiedGroupKFold(n_splits=splits, random_state=5, shuffle = True)
    counter = 0
    
    for counter, (train_index, test_index) in enumerate(tqdm(stratified_group_kfold.split(data, labels, groups), total=5, desc="Processing")):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        if model == 'SVM':
            predictions = run_svm(X_train, X_test, y_train)
        if model =='RF':
            predictions = run_random_forest(X_train, X_test, y_train)
        
        print("fold", counter)
        metrics = get_performance_metrics(y_test, predictions)
      
        print(metrics)

    return metrics 



def get_groups(data, labels, groups, splits): 

    sgkf = StratifiedGroupKFold(n_splits=splits, random_state=5, shuffle = True)

    for i, (train_index, test_index) in enumerate(sgkf.split(data, labels, groups)):
        print(f"Fold {i}:")
    
        # Extract unique group IDs for the training and testing sets
        unique_train_group_ids = np.unique(groups[train_index])
        unique_test_group_ids = np.unique(groups[test_index])
        
        # Extract labels for training and testing sets
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        
        # Calculate the number of positive and negative samples for training set
        train_positive_samples = np.sum(train_labels == 1)
        train_negative_samples = np.sum(train_labels == 0)
        
        # Calculate the number of positive and negative samples for testing set
        test_positive_samples = np.sum(test_labels == 1)
        test_negative_samples = np.sum(test_labels == 0)
    
        print(f"  Train: index={train_index}")
        print(f"         unique groups={unique_train_group_ids}")
        print(f"         Positive samples={train_positive_samples}")
        print(f"         Negative samples={train_negative_samples}")
        
        print(f"  Test:  index={test_index}")
        print(f"         unique groups={unique_test_group_ids}")
        print(f"         Positive samples={test_positive_samples}")
        print(f"         Negative samples={test_negative_samples}")
        
        print("")  # Add an empty line between folds
