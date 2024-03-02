#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from import_and_load import pd, tqdm, features_matrix, labels_matrix

physio_patho_labels = labels_matrix[:,0]
soz_labels = labels_matrix[:,1]

# print(physio_patho_labels.shape)

# Extract Pandas DF with only Physio and Patho Segments

segments_file = pd.read_csv('/Users/soulaimanebentaleb/Desktop/Research/DATASET_MAYO/segments.csv')

# print(segments_file)

segments_df = segments_file[segments_file['category_id'].isin([2,3])]
segments_df = segments_df.reset_index(drop=True)
segments_df = segments_df.drop(columns=['index'])

# print(segments_df)

# Create features and labels lists for the 24 patients and 6 groups of 3

# create indices list for all 24 patients with the indices for physio and patho segments

indices_list = [np.empty((0,))] * 24

for i in tqdm(range(len(segments_df))):        
    p = 0
    while p < 24:
        #if segment is from patient p
        if segments_df['patient_id'][i] == p:
            #append index to matrix
            indices_list[p] = np.append(indices_list[p], [i], axis=0)
            break
        p+=1

# print(len(indices_list))

# create features list with all 24 patients' features

features_list = [None] * 24

for i in tqdm(range(24)):
    features_list[i] = features_matrix[indices_list[i].astype(int)]

# create labels list with all 24 patients' labels

labels_list = [None] * 24

for i in tqdm(range(24)):
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

group_labels_list = [group_1_labels, group_2_labels, group_3_labels, group_4_labels, group_5_labels, group_6_labels]

