from import_and_load import * 
from CV import get_groups, run_CV

data, labels, groups = load_data('Features_Matrix.csv', 'Labels_Matrix.csv')
print(data.shape)
print(groups.shape)
print(labels.shape)
get_groups(data, labels, groups)

metrics = run_CV(data, labels, groups, 'SVM')
