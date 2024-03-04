import numpy as np
from sklearn.decomposition import PCA

# define function to run PCA with the minimum number of components that explain 90% of the variance

def run_PCA(features_train, features_test):
    pca_var = PCA()
    pca_var.fit_transform(features_train)
    exp_var_pca = pca_var.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    for i in cum_sum_eigenvalues:
        if i>=0.9:
            num_components = np.where(cum_sum_eigenvalues==i)[0][0]+1
            break

    pca = PCA(n_components=num_components)
    PCA_train = pca.fit_transform(features_train)
    PCA_test = pca.transform(features_test)
    return PCA_train, PCA_test