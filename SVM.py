from sklearn import svm
import numpy as np

def run_svm(data, labels):

    clf = svm.SVC()

    clf.fit(data, labels)
    predictions = clf.predict(data)

    return predictions

