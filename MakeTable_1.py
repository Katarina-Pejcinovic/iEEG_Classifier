#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

def format_results_to_table(KM_metrics_list, KM_f2_list, SVM_metrics_list, SVM_f2_list, RF_metrics_list, RF_f2_list):
    # Define a new DataFrame for the formatted results
    formatted_results = pd.DataFrame(columns=['Model', 'F2 mean', 'F2 std', 'Accu mean', 'Accu std', 'Precision mean', 'Precision std', 'Recall mean', 'Recall std'])
    
    # Calculate mean and standard deviation for each metric and each model
    for metrics_list, f2_list, model_name in zip(
        [KM_metrics_list, SVM_metrics_list, RF_metrics_list],
        [KM_f2_list, SVM_f2_list, RF_f2_list],
        ['K-Means', 'SVM', 'Random Forest']
    ):
        # Convert metrics list to DataFrame to use pandas functions for mean and std
        df = pd.DataFrame(metrics_list)
        df['F2'] = f2_list
        
        # Calculate mean and std values for all metrics
        means = df.mean()
        stds = df.std()
        
        # Append the results for this model to the formatted_results DataFrame
        formatted_results = formatted_results.append({
            'Model': model_name,
            'F2 mean': means['F2'],
            'F2 std': stds['F2'],
            'Accu mean': means['Accuracy'],  # Replace 'Accuracy' with the correct column name from your metrics
            'Accu std': stds['Accuracy'],    # Replace 'Accuracy' with the correct column name from your metrics
            'Precision mean': means['Precision'],  # Replace 'Precision' with the correct column name from your metrics
            'Precision std': stds['Precision'],    # Replace 'Precision' with the correct column name from your metrics
            'Recall mean': means['Recall'],        # Replace 'Recall' with the correct column name from your metrics
            'Recall std': stds['Recall']           # Replace 'Recall' with the correct column name from your metrics
        }, ignore_index=True)
    
    return formatted_results



#results_table = format_results_to_table(KM_metrics_list, KM_f2_list, SVM_metrics_list, SVM_f2_list, RF_metrics_list, RF_f2_list)

#print(results_table)

