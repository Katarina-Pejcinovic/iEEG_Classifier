from import_and_load import * 
from CV import get_groups, run_CV
from pca_analysis import * 
from train_test_split import *
from get_mean_variance import get_mean_variance
from manual_CV import * 
from MakeTable_1 import *

'''#separate data into 5 groups'''

features_matrix = np.genfromtxt('/Users/andresmichel/Desktop/Features Matrix.csv', delimiter=',')
labels_matrix = np.genfromtxt('/Users/andresmichel/Desktop/Labels Matrix.csv', delimiter=',')


# Extract Pandas DF with only Physio and Patho Segments
segments_file = pd.read_csv('/Users/andresmichel/Desktop/DATASET_MAYO/segments.csv')

# print(segments_file)
segments_df = segments_file[segments_file['category_id'].isin([2,3])]
segments_df = segments_df.reset_index(drop=True)
segments_df = segments_df.drop(columns=['index'])

# create indices list for all 24 patients with the indices for physio and patho segments
physio_patho_labels = labels_matrix[:,0]
print(physio_patho_labels)
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


'''create folds'''
#fold 1 = 0, 1, 2 (testing set)
fold_1_data =[features_list[0], features_list[1], features_list[2]]
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

#fold 5 = 7, 21
fold_5_data = [features_list[7], features_list[21]]
fold_5_labels = [labels_list[7], labels_list[21]]

data_folds = [fold_2_data, fold_3_data, fold_4_data, fold_5_data]
label_folds = [fold_2_labels, fold_3_labels, fold_4_labels, fold_5_labels]

print("made folds")

data_folds, label_folds = get_np_folds(data_folds, label_folds)
#print("len data", len(data_folds), len(label_folds))

KM_metrics_list, KM_f2_list, SVM_metrics_list, SVM_f2_list, RF_metrics_list, RF_f2_list  = manual_cross_val(data_folds, label_folds)
# print("KM", '\n', KM_metrics_list, KM_f2_list, )
# print("SVM", '\n', SVM_metrics_list, SVM_f2_list, )
# print("RF", RF_metrics_list, RF_f2_list)

KM_f2_mean, KM_f2_var = get_mean_variance(KM_f2_list)
SVM_f2_mean, SVM_f2_var = get_mean_variance(SVM_f2_list)
RF_f2_mean, RF_f2_var = get_mean_variance(RF_f2_list)

# print("K-Means F2 mean : ", KM_f2_mean)
# print("K-Means F2 variance : ", KM_f2_var)
# print("SVM F2 mean : ", SVM_f2_mean)
# print("SVM F2 variance : ", SVM_f2_var)
# print("Random Forest F2 mean : ", RF_f2_mean)
# print("Random Forest F2 variance : ", RF_f2_var)

table = format_results_to_table(KM_metrics_list, KM_f2_list, SVM_metrics_list, SVM_f2_list, RF_metrics_list, RF_f2_list)
print(table)