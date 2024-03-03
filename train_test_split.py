import pickle as pkl
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from numpy import genfromtxt



def train_test_split(data, labels, groups):
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=2)

    for train_index, test_index in gss.split(data, labels, groups):
        print("Train:", train_index)
        print("Test:", test_index)
        print("  Unique groups:", np.unique(groups[train_index]))

        print("\nTest:")
        print("  Index:", test_index)
        print("  Unique groups:", np.unique(groups[test_index]))

        # Get numpy arrays for training and testing data
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        train_groups, test_groups = groups[train_index], groups[test_index]

        return train_data, test_data, train_labels, test_labels, train_groups, test_groups

    return 
    
    