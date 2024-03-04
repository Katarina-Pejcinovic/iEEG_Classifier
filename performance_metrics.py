from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score
import numpy as np

# define function to calculate performance metrics (accuracy, precision, recall, and F2 score)

def get_performance_metrics(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    if accuracy < 0.5:
        for i in range(len(pred_labels)):
            if pred_labels[i] == 1:
                pred_labels[i] = 0
            elif pred_labels[i] == 0:
                pred_labels[i] = 1
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f2 = fbeta_score(true_labels, pred_labels, beta=2)
    return np.array([accuracy, precision, recall, f2])