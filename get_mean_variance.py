#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def get_mean_variance(f2_list):
    mean = np.mean(np.array([i for i in f2_list if i is not None]))
    var = np.var(np.array([i for i in f2_list if i is not None]))
    return mean*100, var*100

