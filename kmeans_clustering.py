from sklearn.cluster import KMeans

# define function to run K-Means

def run_KMeans(train_data, test_data):
    kmeans = KMeans(n_clusters=2, random_state=1, n_init='auto')
    kmeans.fit(train_data)
    pred_labels = kmeans.predict(test_data)
    return pred_labels