import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score  # You can replace this with the appropriate metric for your problem
from sklearn.ensemble import RandomForestClassifier  # You can replace this with the model of your choice
import pickle as pkl
import pandas as pd 

def manaul_cross_val(data_folds, label_folds):

    num_folds = len(data_folds)
    accuracy_scores = []

    # Perform 4-fold cross-validation
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(range(num_folds)):

        # Select training and testing data for this fold
        train_data, test_data = np.concatenate([data_folds[i] for i in train_index]), np.concatenate([data_folds[i] for i in test_index])
        train_labels, test_labels = np.concatenate([label_folds[i] for i in train_index]), np.concatenate([label_folds[i] for i in test_index])

        # Initialize and train the model
        model = RandomForestClassifier()  # You can replace this with the model of your choice
        model.fit(train_data, train_labels)

        # Make predictions on the test set
        predictions = model.predict(test_data)

        # Evaluate the model and store the accuracy score
        accuracy = accuracy_score(test_labels, predictions)
        accuracy_scores.append(accuracy)

    return accuracy_scores

#separate data into 5 groups

#features_matrix = np.genfromtxt(file_path_data, delimiter=',')
labels_matrix = np.genfromtxt('Features_Matrix.csv', delimiter=',')

with open('feature_data.pkl', 'rb') as f:
    features_matrix = pkl.load(f)


# Extract Pandas DF with only Physio and Patho Segments
segments_file = pd.read_csv('/Users/katarinapejcinovic/Downloads/DATASET_MAYO/segments.csv')

# print(segments_file)
segments_df = segments_file[segments_file['category_id'].isin([2,3])]
segments_df = segments_df.reset_index(drop=True)
segments_df = segments_df.drop(columns=['index'])

# create indices list for all 24 patients with the indices for physio and patho segments
physio_patho_labels = labels_matrix[:,0]
soz_labels = labels_matrix[:,1]

indices_list = [np.empty((0,))] * 24

for i in range(len(segments_df)):        
    p = 0
    while p < 24:
        #if segment is from patient p
        if segments_df['patient_id'][i] == p:
            #append index to matrix
            indices_list[p] = np.append(indices_list[p], [i], axis=0)
            break
        p+=1


# create features list with all 24 patients' features
features_list = [None] * 24

for i in range(24):
    features_list[i] = features_matrix[indices_list[i].astype(int)]

# create labels list with all 24 patients' labels
labels_list = [None] * 24

for i in (range(24)):
    labels_list[i] = physio_patho_labels[indices_list[i].astype(int)]

# create a list with the features from patients in each group

group_1_features = [features_list[0], features_list[18], features_list[21]]
group_2_features = [features_list[1], features_list[9], features_list[19]]
group_3_features = [features_list[2], features_list[5], features_list[16]]
group_4_features = [features_list[3], features_list[4], features_list[23]]
group_5_features = [features_list[6], features_list[7], features_list[8]]
group_6_features = [features_list[14], features_list[17], features_list[20]]

group_features_list = [group_1_features, group_2_features, group_3_features, group_4_features, group_5_features, group_6_features]

# create a list with the labels from patients in each group

group_1_labels = [labels_list[0], labels_list[18], labels_list[21]]
group_2_labels = [labels_list[1], labels_list[9], labels_list[19]]
group_3_labels = [labels_list[2], labels_list[5], labels_list[16]]
group_4_labels = [labels_list[3], labels_list[4], labels_list[23]]
group_5_labels = [labels_list[6], labels_list[7], labels_list[8]]
group_6_labels = [labels_list[14], labels_list[17], labels_list[20]]

'''create folds'''
#fold 1 = 0, 1, 2
fold_1_data = [features_list[0], features_list[1], features_list[2]]
fold_1_labels = [labels_list[0], labels_list[1], labels_list[2]]

#fold 2 = 3, 17, 23
fold_2_data = [features_list[3], features_list[17], features_list[23]]
fold_2_labels = [labels_list[3], labels_list[17], labels_list[23]]

#fold 3 = 4,8 ,16, 18
fold_3_data = [features_list[4], features_list[8], features_list[16], features_list[18]]
fold_3_labels = [labels_list[4], labels_list[8], labels_list[16], labels_list[18]]

#fold 4 = 5, 14, 20 
fold_4_data = [features_list[5], features_list[14], features_list[20]]
fold_4_labels = [labels_list[5], labels_list[14], labels_list[20]]

#fold 5 = 7, 21 (testing)
fold_5_data = [features_list[7], features_list[21]]
fold_5_labels = [labels_list[7], labels_list[21]]

data_folds = [fold_1_data, fold_2_data, fold_3_data, fold_4_data]
label_folds = [fold_1_labels, fold_2_labels, fold_3_labels, fold_4_labels]

accuracy_scores = manaul_cross_val(data_folds, label_folds)
print("Accuracy scores for each fold:", accuracy_scores)
print("Mean accuracy:", np.mean(accuracy_scores))
