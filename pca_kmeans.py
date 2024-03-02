#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scaling_features import scale_features
from pca_analysis import run_PCA
from kmeans_clustering import run_KMeans

# define function to run PCA + K-Means and output the predicted labels

def PCA_KMeans(features):
    scaled_features = scale_features(features)
    X_PCA = run_PCA(scaled_features)
    pred_labels = run_KMeans(X_PCA)

