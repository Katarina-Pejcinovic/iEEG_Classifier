#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.stats as stats

def scale_features(features):
    scaled_features = stats.zscore(features)
    clean_scaled_features = np.delete(scaled_features, np.isnan(scaled_features[0]), axis=1)
    return clean_scaled_features

