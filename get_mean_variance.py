#!/usr/bin/env python
# coding: utf-8

import numpy as np

def get_mean_variance(f2_list):
    mean = np.mean(np.array([i for i in f2_list if i is not None]))
    var = np.var(np.array([i for i in f2_list if i is not None]))
    return mean, var
