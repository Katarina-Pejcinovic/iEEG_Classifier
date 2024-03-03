
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

# Load Features + Labels Datasets

features_matrix = np.genfromtxt('Features_Matrix.csv', delimiter=',')
labels_matrix = np.genfromtxt('Labels_Matrix.csv', delimiter=',')

# Print Dimensions of Features and Labels Matrices

# print(labels_matrix)
# print(features_matrix.shape)
# print(labels_matrix.shape)

physio_patho_labels = labels_matrix[:,0]
soz_labels = labels_matrix[:,1]

# print(physio_patho_labels.shape)

# Extract Pandas DF with only Physio and Patho Segments
segments_file = pd.read_csv('/Users/katarinapejcinovic/Downloads/DATASET_MAYO/segments.csv')

# print(segments_file)

segments_df = segments_file[segments_file['category_id'].isin([2,3])]
segments_df = segments_df.reset_index(drop=True)
segments_df = segments_df.drop(columns=['index'])

#print("segments", segments_df)

# Create features and labels lists for the 24 patients and 6 groups of 3

# create indices list for all 24 patients with the indices for physio and patho segments

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

print("len indices", len(indices_list))
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
print("features list", len(features_list[0]))
# create a list with the labels from patients in each group

group_1_labels = [labels_list[0], labels_list[18], labels_list[21]]
group_2_labels = [labels_list[1], labels_list[9], labels_list[19]]
group_3_labels = [labels_list[2], labels_list[5], labels_list[16]]
group_4_labels = [labels_list[3], labels_list[4], labels_list[23]]
group_5_labels = [labels_list[6], labels_list[7], labels_list[8]]
group_6_labels = [labels_list[14], labels_list[17], labels_list[20]]

group_labels_list = [group_1_labels, group_2_labels, group_3_labels, group_4_labels, group_5_labels, group_6_labels]

'''
# define function to scale features using z-score

def scale_features(features):
    scaled_features = stats.zscore(features)
    clean_scaled_features = np.delete(scaled_features, np.isnan(scaled_features[0]), axis=1)
    return clean_scaled_features

# define function to calculate performance metrics (accuracy, precision, recall, and F2 score)

def get_performance_metrics(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    if accuracy < 0.5:
        for i in range(len(pred_labels)):
            if pred_labels[i] == 1:
                pred_labels[i] = 0
            elif pred_labels[i] == 0:
                pred_labels[i] = 1
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f2 = fbeta_score(true_labels, pred_labels, beta=2)
    return np.array([accuracy, precision, recall, f2])

# define function to calculate the mean and variance of the F2 scores in a list (will be used after we run CV and get the F2 score from each fold)

def get_mean_variance(f2_list):
    mean = np.mean(np.array([i for i in f2_list if i is not None]))
    var = np.var(np.array([i for i in f2_list if i is not None]))
    return mean*100, var*100


# Script to visualize PCA variance and decide number of components that explain 90% of the variance

pca_all = PCA()

# Determine transformed features
X_PCA = pca_all.fit_transform(scale_features(features_matrix))

# Determine explained variance using explained_variance_ration_ attribute
exp_var_pca_all = pca_all.explained_variance_ratio_

# Cumulative sum of eigenvalues for visualizing the cumulative variance explained by each principal component
cum_sum_eigenvalues_all = np.cumsum(exp_var_pca_all)

# Create the visualization plot
# plt.figure(figsize = (15,5))
# plt.subplot(1, 2, 1)
# plt.plot(exp_var_pca_all*100)
# plt.title('Individual Variance Explained by Each Principal Component')
# plt.ylabel('Variance (%)')
# plt.xlabel('Principal Component Index')
# plt.subplot(1, 2, 2)
# plt.plot(cum_sum_eigenvalues_all*100)
# plt.title('Cumulative Variance Explained by Principal Components')
# plt.ylabel('Variance (%)')
# plt.xlabel('Principal Component Index')
# plt.tight_layout()

#Determine the minimum number of components to explain 90% of the variance
for i in cum_sum_eigenvalues_all:
    if i>=0.9:
        num_comp_all = np.where(cum_sum_eigenvalues_all==i)[0][0]+1
        # print("We need " + str(num_comp_all) + " principal components to explain 90% of the variance.")
        break
        
# print("Variance explained by " + str(num_comp_all) + " components = " + str(cum_sum_eigenvalues_all[num_comp_all-1] * 100) + '%')


# define function to run PCA

def run_PCA(features, num_components=num_comp_all):
    pca = PCA(n_components=num_components)
    X_PCA = pca.fit_transform(features)
    return X_PCA

# define function to run K-Means

def run_KMeans(data):
    np.random.seed(0)
    kmeans = KMeans(n_clusters=2, random_state=1, n_init='auto')
    kmeans.fit(power_transform(data))
    pred_labels = kmeans.predict(power_transform(data))
    return pred_labels

# define function to run PCA + K-Means and output the predicted labels

def PCA_KMeans(features):
    scaled_features = scale_features(features)
    X_PCA = run_PCA(scaled_features)
    pred_labels = run_KMeans(X_PCA)
'''