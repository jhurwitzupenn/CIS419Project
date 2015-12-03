"""
=====================================
Train Shortest Path Classifiers
=====================================

Author: Reuben Abrahamn, Nivedita Sankar, Jordan Hurwitz

Adapted from scikit_learn documentation.

"""
print(__doc__)

import numpy as np
from sklearn import svm, datasets, grid_search, tree
from sklearn.metrics import accuracy_score
import time

def trainClassifiers(X, Y):
    C = 0.01
    gam = 0.01
    # grid search over these to find parameters
    svm_rbf_model = svm.SVC(C = C, kernel='rbf', gamma=gam)
    svm_sig_model = svm.SVC(C = C, kernel='sigmoid', gamma=gam)
    dtree_model = tree.DecisionTreeClassifier(max_depth=3)

    # fit the models
    print "Training the SVM RBF model..."
    rbf_train_time = time.clock()
    svm_rbf_model.fit(X, Y)
    rbf_train_time = time.clock() - rbf_train_time

    print "Training the SVM Sigmoid model..."
    sig_train_time = time.clock()
    svm_sig_model.fit(X, Y)
    sig_train_time = time.clock() - sig_train_time

    print "Training the DecisionTree model..."
    dtree_train_time = time.clock()
    dtree_model.fit(X, Y)
    dtree_train_time = time.clock() - dtree_train_time

    print "SVM RBF Training time: " + str(rbf_train_time)
    print "SVM Sigmoid Training time: " + str(sig_train_time)
    print "DecisionTree Training time: " + str(dtree_train_time)

    return svm_rbf_model, svm_sig_model, dtree_model

