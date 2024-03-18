from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import accuracy_score

# define function to run K-Means

def run_KMeans(pca_train_data, pca_test_data, train_data, test_data):
    kmeans = KMeans(n_clusters=2, random_state=1, n_init='auto')
    kmeans.fit(pca_train_data)
    pred_labels = kmeans.predict(pca_test_data)

    index0 = np.where(pred_labels == 0)[0]
    index1 = np.where(pred_labels == 1)[0]

    median_0 = np.array([])
    median_1 = np.array([])

    for i in [-22, -24, -25, -26]:   
        bp_0 = np.array([])
        for f in train_data[index0]:
            bp_0 = np.append(bp_0, f[i])
        median_0 = np.append(median_0, np.median(bp_0))

        bp_1 = np.array([])
        for f in test_data[index1]:
            bp_1 = np.append(bp_1, f[i])
        median_1 = np.append(median_1, np.median(bp_1))

    if np.sum(median_0) < np.sum(median_1):
        for i in range(len(pred_labels)):
            if pred_labels[i] == 1:
                pred_labels[i] = 0
            elif pred_labels[i] == 0:
                pred_labels[i] = 1

    return pred_labels