from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import numpy as np
import time
# from sklearn.externals.six import StringIO
# import pydot

"""
=====================================
Train Shortest Path Classifiers
=====================================

Author: Reuben Abrahamn, Nivedita Sankar, Jordan Hurwitz

Adapted from scikit_learn documentation.

"""
print(__doc__)


def SVM(X, Y, Xtest, Ytest):
    # grid search over these to find parameters
    CList = [.001, .003, .01, .03, .1, .3, 1, 3, 6, 10, 15, 30, 40]
    gammaList = [.001, .003, .01, .03, .1, .3, 1, 2, 3, 4, 5, 6, 7]
    param_grid = [{'C': CList,
                   'gamma': gammaList,
                   'kernel': ['rbf', 'sigmoid', 'linear']}]
    # grid search over these to find parameters
    rbf_grid = GridSearchCV(SVC(), param_grid=param_grid)
    # fit the models
    rbf_grid.fit(X, Y)

    print "Predicting with the SVM RBF model..."
    rbf_predict_time = time.time()
    Ypred_rbf = rbf_grid.predict(X)
    print Y
    print Ypred_rbf
    rbf_predict_time = time.time() - rbf_predict_time

    rbf_accuracy = metrics.accuracy_score(Ytest, Ypred_rbf)
    rbf_precision = metrics.precision_score(Ytest, Ypred_rbf, average='binary')
    rbf_recall = metrics.recall_score(Ytest, Ypred_rbf, average='binary')

    print "SVM RBF Prediction time: " + str(rbf_predict_time)
    print "SVM RBF Accuracy Score: " + str(rbf_accuracy)
    print "SVM RBF Precision Score: " + str(rbf_precision)
    print "SVM RBF Recall Score: " + str(rbf_recall)


def SVMSig(X, Y, Xtest, Ytest):
    C = 0.01
    gam = 0.01
    # grid search over these to find parameters
    svm_sig_model = svm.SVC(C=C, kernel='sigmoid', gamma=gam)

    print "Training the SVM Sigmoid model..."
    sig_train_time = time.time()
    svm_sig_model.fit(X, Y)
    sig_train_time = time.time() - sig_train_time

    print "SVM Sigmoid Training time: " + str(sig_train_time)

    print "Predicting with the SVM Sigmoid model..."
    sig_predict_time = time.time()
    Ypred_sig = svm_sig_model.predict(X)
    sig_predict_time = time.time() - sig_predict_time

    sig_accuracy = metrics.accuracy_score(Ytest, Ypred_sig)
    sig_precision = metrics.precision_score(Ytest, Ypred_sig, average='binary')
    sig_recall = metrics.recall_score(Ytest, Ypred_sig, average='binary')

    print "SVM Sigmoid Prediction time: " + str(sig_predict_time)
    print "SVM Sigmoid Accuracy Score: " + str(sig_accuracy)
    print "SVM Sigmoid Precision Score: " + str(sig_precision)
    print "SVM Sigmoid Recall Score: " + str(sig_recall)


def DTree(X, Y, Xtest, Ytest):
    print "Training the DecisionTree model..."

    # dot_data = StringIO()
    # tree.export_graphviz(dtree_model, out_file=dot_data)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("../dtree.pdf")

    param_grid = {'max_depth': np.arange(1, 15)}

    tree_grid = GridSearchCV(DecisionTreeClassifier(), param_grid)
    tree_grid.fit(X, Y)

    print("The best parameters are %s with a score of %0.2f"
          % (tree_grid.best_params_, tree_grid.best_score_))

    print "Predicting with the Decision Tree model..."
    dtree_predict_time = time.time()
    Ypred_dtree = tree_grid.predict(X)
    dtree_predict_time = time.time() - dtree_predict_time

    dtree_accuracy = metrics.accuracy_score(Ytest, Ypred_dtree)
    dt_precision = metrics.precision_score(Ytest, Ypred_dtree, average='binary')
    dtree_recall = metrics.recall_score(Ytest, Ypred_dtree, average='binary')

    print "Decision Tree Prediction time: " + str(dtree_predict_time)
    print "Decision Tree Accuracy Score: " + str(dtree_accuracy)
    print "Decision Tree Precision Score: " + str(dt_precision)
    print "Decision Tree Recall Score: " + str(dtree_recall)


def makePredictions(X, Y, rbf_model, sig_model, dtree_model):
    print "Predicting with the SVM RBF model..."
    rbf_predict_time = time.time()
    Ypred_rbf = rbf_model.predict(X)
    rbf_predict_time = time.time() - rbf_predict_time

    print "Predicting with the SVM Sigmoid model..."
    sig_predict_time = time.time()
    Ypred_sig = sig_model.predict(X)
    sig_predict_time = time.time() - sig_predict_time

    print "Predicting with the SVM Sigmoid model..."
    dtree_predict_time = time.time()
    Ypred_dtree = sig_model.predict(X)
    dtree_predict_time = time.time() - dtree_predict_time

    rbf_accuracy = metrics.accuracy_score(Y, Ypred_rbf)
    sig_accuracy = metrics.accuracy_score(Y, Ypred_sig)
    dtree_accuracy = metrics.accuracy_score(Y, Ypred_dtree)
    rbf_precision = metrics.precision_score(Y, Ypred_rbf, average='macro')
    sig_precision = metrics.precision_score(Y, Ypred_sig, average='macro')
    dtree_precision = metrics.precision_score(Y, Ypred_dtree, average='macro')
    rbf_recall = metrics.recall_score(Y, Ypred_rbf, average='macro')
    sig_recall = metrics.recall_score(Y, Ypred_sig, average='macro')
    dtree_recall = metrics.recall_score(Y, Ypred_dtree, average='macro')

    print "SVM RBF Prediction time: " + str(rbf_predict_time)
    print "SVM Sigmoid Prediction time: " + str(sig_predict_time)
    print "DecisionTree Prediction time: " + str(dtree_predict_time)

    print "SVM RBF Accuracy Score: " + str(rbf_accuracy)
    print "SVM Sigmoid Accuracy Score: " + str(sig_accuracy)
    print "DecisionTree Accuracy Score: " + str(dtree_accuracy)

    print "SVM RBF Precision Score: " + str(rbf_precision)
    print "SVM Sigmoid Precision Score: " + str(sig_precision)
    print "DecisionTree Precision Score: " + str(dtree_precision)

    print "SVM RBF Recall Score: " + str(rbf_recall)
    print "SVM Sigmoid Recall Score: " + str(sig_recall)
    print "DecisionTree Recall Score: " + str(dtree_recall)
