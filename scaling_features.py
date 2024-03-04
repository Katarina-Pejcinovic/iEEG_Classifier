#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    clean_scaled_features = np.delete(scaled_features, np.isnan(scaled_features[0]), axis=1)
    return clean_scaled_features