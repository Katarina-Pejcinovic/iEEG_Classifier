#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# define function to run K-Means
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import power_transform

def run_KMeans(data):
    np.random.seed(0)
    kmeans = KMeans(n_clusters=2, random_state=1, n_init='auto')
    kmeans.fit(power_transform(data))
    pred_labels = kmeans.predict(power_transform(data))
    return pred_labels

