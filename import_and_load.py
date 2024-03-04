#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import power_transform
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score
import pickle as pkl

# Load Features + Labels Datasets
def load_data(file_path_data, file_path_labels):

    #features_matrix = np.genfromtxt(file_path_data, delimiter=',')
    labels_matrix = np.genfromtxt(file_path_labels, delimiter=',')

    with open('feature_data.pkl', 'rb') as f:
        features_matrix = pkl.load(f)

    #print(features_matrix.shape)

    # Extract Pandas DF with only Physio and Patho Segments
    segments_file = pd.read_csv('/Users/katarinapejcinovic/Downloads/DATASET_MAYO/segments.csv')

    # print(segments_file)
    segments_df = segments_file[segments_file['category_id'].isin([2,3])]
    segments_df = segments_df.reset_index(drop=True)
    segments_df = segments_df.drop(columns=['index'])

    #print("segments", segments_df)

    # Create features and labels lists for the 24 patients and 6 groups of 3

    # create indices list for all 24 patients with the indices for physio and patho segments
    physio_patho_labels = labels_matrix[:,0]
    soz_labels = labels_matrix[:,1]

    indices_list = [np.empty((0,))] * 24

    for i in tqdm(range(len(segments_df))):        
        p = 0
        while p < 24:
            #if segment is from patient p
            if segments_df['patient_id'][i] == p:
                #append index to matrix
                indices_list[p] = np.append(indices_list[p], [i], axis=0)
                break
            p+=1

    #print("len indices", len(indices_list))
    #print(indices_list)

    # create features list with all 24 patients' features
    features_list = [None] * 24

    for i in tqdm(range(24)):
        features_list[i] = features_matrix[indices_list[i].astype(int)]

    # create labels list with all 24 patients' labels
    labels_list = [None] * 24

    for i in tqdm(range(24)):
        labels_list[i] = physio_patho_labels[indices_list[i].astype(int)]

    # create a list with the features from patients in each group

    group_1_features = [features_list[0], features_list[18], features_list[21]]
    group_2_features = [features_list[1], features_list[9], features_list[19]]
    group_3_features = [features_list[2], features_list[5], features_list[16]]
    group_4_features = [features_list[3], features_list[4], features_list[23]]
    group_5_features = [features_list[6], features_list[7], features_list[8]]
    group_6_features = [features_list[14], features_list[17], features_list[20]]

    group_features_list = [group_1_features, group_2_features, group_3_features, group_4_features, group_5_features, group_6_features]

    # create a list with the labels from patients in each group

    group_1_labels = [labels_list[0], labels_list[18], labels_list[21]]
    group_2_labels = [labels_list[1], labels_list[9], labels_list[19]]
    group_3_labels = [labels_list[2], labels_list[5], labels_list[16]]
    group_4_labels = [labels_list[3], labels_list[4], labels_list[23]]
    group_5_labels = [labels_list[6], labels_list[7], labels_list[8]]
    group_6_labels = [labels_list[14], labels_list[17], labels_list[20]]

    group_labels_list = [group_1_labels, group_2_labels, group_3_labels, group_4_labels, group_5_labels, group_6_labels]
    for i in range(24):
        counter_one = 0 
        counter_zero = 0
        labels = labels_list[i]
        for r in range (len(labels)):
            if (labels[r] == 0):
                counter_zero +=1
            else:
                counter_one +=1

        print("patient", i, '\n', counter_one, counter_zero)
        

    patients = [0, 1, 2, 3, 4, 5, 7, 8, 14, 16, 17, 18, 20, 21, 23]
    groups = []
    for patient in patients:
        #print("patient id", patient)

        repeat = np.repeat(patient, len(labels_list[patient]))
        #print("repeated", repeat)
        for i in range(len(labels_list[patient])):
            groups.append(patient)

       # print("size", len(groups))

    groups = np.array(groups)
    return features_matrix, labels_matrix[:,0], groups

