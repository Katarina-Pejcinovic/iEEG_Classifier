

from sklearn.decomposition import PCA
from scaling_features import scale_features
import numpy as np 

# Script to visualize PCA variance and decide number of components that explain 90% of the variance

pca_all = PCA()
features_matrix = np.genfromtxt('Features_Matrix.csv', delimiter=',') 

# Determine transformed features
X_PCA = pca_all.fit_transform(scale_features(features_matrix))

# Determine explained variance using explained_variance_ration_ attribute
exp_var_pca_all = pca_all.explained_variance_ratio_

# Cumulative sum of eigenvalues for visualizing the cumulative variance explained by each principal component
cum_sum_eigenvalues_all = np.cumsum(exp_var_pca_all)

# Create the visualization plot
# plt.figure(figsize = (15,5))
# plt.subplot(1, 2, 1)
# plt.plot(exp_var_pca_all*100)
# plt.title('Individual Variance Explained by Each Principal Component')
# plt.ylabel('Variance (%)')
# plt.xlabel('Principal Component Index')
# plt.subplot(1, 2, 2)
# plt.plot(cum_sum_eigenvalues_all*100)
# plt.title('Cumulative Variance Explained by Principal Components')
# plt.ylabel('Variance (%)')
# plt.xlabel('Principal Component Index')
# plt.tight_layout()

#Determine the minimum number of components to explain 90% of the variance
for i in cum_sum_eigenvalues_all:
    if i>=0.9:
        num_comp_all = np.where(cum_sum_eigenvalues_all==i)[0][0]+1
        # print("We need " + str(num_comp_all) + " principal components to explain 90% of the variance.")
        break
        
print("num eigenvales", num_comp_all)
# print("Variance explained by " + str(num_comp_all) + " components = " + str(cum_sum_eigenvalues_all[num_comp_all-1] * 100) + '%')

# define function to run PCA

def run_PCA(features, num_components=num_comp_all):
    pca = PCA(n_components=num_components)
    X_PCA = pca.fit_transform(features)
    return X_PCA

