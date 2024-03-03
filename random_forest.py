#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier

def run_random_forest(X_train, X_test, y_train):
    """
    Parameters:
    - X_train: numpy array, the training feature matrix.
    - X_test: numpy array, the testing feature matrix.
    - y_train: numpy array, the labels corresponding to each row in the training feature matrix.

    Returns:
    - predictions: numpy array, the predicted labels for the test data.
    """

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    rf_model.fit(X_train, y_train)
    
    predictions = rf_model.predict(X_test)

    return predictions


# In[ ]:




