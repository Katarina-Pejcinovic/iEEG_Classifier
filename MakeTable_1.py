#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
#add to test_run.py
#from MakeTable import format_results_to_table

def format_results_to_table(KM_metrics_list, KM_f2_list, SVM_metrics_list, SVM_f2_list, RF_metrics_list, RF_f2_list):

    KM_df = pd.DataFrame(KM_metrics_list)
    KM_df['F2'] = KM_f2_list
    KM_df['Model'] = 'K-Means'
    
    SVM_df = pd.DataFrame(SVM_metrics_list)
    SVM_df['F2'] = SVM_f2_list
    SVM_df['Model'] = 'SVM'
    
    RF_df = pd.DataFrame(RF_metrics_list)
    RF_df['F2'] = RF_f2_list
    RF_df['Model'] = 'Random Forest'
    
    # Combine all dataframes into one
    results_df = pd.concat([KM_df, SVM_df, RF_df])
    results_df = results_df.reset_index(drop=True)
    return results_df


#results_table = format_results_to_table(KM_metrics_list, KM_f2_list, SVM_metrics_list, SVM_f2_list, RF_metrics_list, RF_f2_list)

#print(results_table)

