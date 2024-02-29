import pickle as pkl
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import genfromtxt

# my_data = genfromtxt('Features_Matrix.csv', delimiter=',')

# with open('feature_data.pkl', 'wb') as file: 
#     # A new file will be created 
#     pkl.dump(my_data, file) 

my_labels = genfromtxt('Labels_Matrix.csv', delimiter = ',')

# Split the data into training and testing sets
with open('feature_data.pkl', 'rb') as f:
    feat_data = pkl.load(f)

print(feat_data.shape)
print(my_labels)



X_train, X_test, y_train, y_test = train_test_split(feat_data, my_labels, test_size=0.2, random_state=42)
