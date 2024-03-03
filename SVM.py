from sklearn import svm
import numpy as np

def run_svm(data, test, labels):

    clf = svm.SVC()

    clf.fit(data, labels)
    predictions = clf.predict(test)

    return predictions

