import numpy as np 
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
import pickle as pkl




# ica = FastICA()
# fit = ica.fit_transform(X = feat_data)

##PCA 

pca = PCA(n_components= 0.8)
X_train_pca = pca.fit_transform(feat_data)
X_test_pca = pca.transform(feat_data)

with open('pca_train.pkl', 'wb') as file: 
    # A new file will be created 
    pkl.dump(X_train_pca, file) 

