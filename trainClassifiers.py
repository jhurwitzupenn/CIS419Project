from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
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


def SVM(X, Y, XTest, YTest):
    print '-----------------------------------------------------'
    # grid search over these to find parameters
    CList = [.001, .003, .01, .03, .1, .3, 1, 3, 6, 10, 15, 30, 40]
    gammaList = [.001, .003, .01, .03, .1, .3, 1, 2, 3, 4, 5, 6, 7]
    param_grid = [{'C': CList,
                   'gamma': gammaList,
                   'kernel': ['rbf', 'sigmoid', 'linear']}]
    # grid search over these to find parameters
    # rbf_grid = GridSearchCV(SVC(probability=True), param_grid=param_grid)
    rbf_grid = SVC(C=500, gamma=0.1, probability=True)
    # fit the models
    rbf_grid.fit(X, Y)

    # print("The best parameters are %s with a score of %0.2f"
    #       % (rbf_grid.best_params_, rbf_grid.best_score_))

    print "Computing training statistics"
    rbf_predict_time_training = time.time()
    Ypred_rbf_training = rbf_grid.predict(X)
    rbf_predict_time_training = time.time() - rbf_predict_time_training

    rbf_accuracy_training = metrics.accuracy_score(Y, Ypred_rbf_training)
    rbf_precision_training = metrics.precision_score(Y, Ypred_rbf_training,
                                                     average='binary')
    rbf_recall_training = metrics.recall_score(Y, Ypred_rbf_training,
                                               average='binary')

    print "SVM RBF training prediction time: " + str(rbf_predict_time_training)
    print "SVM RBF training accuracy Score: " + str(rbf_accuracy_training)
    print "SVM RBF training precision Score: " + str(rbf_precision_training)
    print "SVM RBF training recall Score: " + str(rbf_recall_training)

    print "Computing testing statistics"
    rbf_predict_time_test = time.time()
    Ypred_rbf_test = rbf_grid.predict(XTest)
    rbf_predict_time_test = time.time() - rbf_predict_time_test

    rbf_accuracy_test = metrics.accuracy_score(YTest, Ypred_rbf_test)
    rbf_precision_test = metrics.precision_score(YTest, Ypred_rbf_test,
                                                 average='binary')
    rbf_recall_test = metrics.recall_score(YTest, Ypred_rbf_test,
                                           average='binary')

    print "SVM RBF test prediction time: " + str(rbf_predict_time_test)
    print "SVM RBF test accuracy Score: " + str(rbf_accuracy_test)
    print "SVM RBF test precision Score: " + str(rbf_precision_test)
    print "SVM RBF test recall Score: " + str(rbf_recall_test)

    print "Creating ROC curve"
    y_true = YTest
    y_score = rbf_grid.predict_proba(XTest)
    fprSVM, trpSVM, _ = metrics.roc_curve(y_true=y_true,
                                          y_score=y_score[:, 0],
                                          pos_label=0)
    plt.plot(fprSVM, trpSVM, 'b-', label='SVM')


def DTree(X, Y, XTest, YTest):
    print '-----------------------------------------------------'
    # dot_data = StringIO()
    # tree.export_graphviz(dtree_model, out_file=dot_data)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("../dtree.pdf")

    # param_grid = {'max_depth': np.arange(1, 15)}

    # tree_grid = GridSearchCV(DecisionTreeClassifier(), param_grid)
    tree_grid = DecisionTreeClassifier(max_depth=3)
    tree_grid.fit(X, Y)
    export_graphviz(tree_grid, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtreevis.pdf")

    # print("The best parameters are %s with a score of %0.2f"
    #       % (tree_grid.best_params_, tree_grid.best_score_))

    print "Computing training statistics"
    dtree_predict_time_training = time.time()
    Ypred_dtree_training = tree_grid.predict(X)
    dtree_predict_time_training = time.time() - dtree_predict_time_training

    dtree_accuracy_training = metrics.accuracy_score(Y, Ypred_dtree_training)
    dt_precision_training = metrics.precision_score(Y, Ypred_dtree_training,
                                                    average='binary')
    dtree_recall_training = metrics.recall_score(Y, Ypred_dtree_training,
                                                 average='binary')

    print "DT training prediction time: " + str(dtree_predict_time_training)
    print "DT training accuracy Score: " + str(dtree_accuracy_training)
    print "DT training precision Score: " + str(dt_precision_training)
    print "DT training recall Score: " + str(dtree_recall_training)

    print "Computing testing statistics"
    dtree_predict_time_test = time.time()
    Ypred_dtree_test = tree_grid.predict(XTest)
    dtree_predict_time_test = time.time() - dtree_predict_time_test

    dtree_accuracy_test = metrics.accuracy_score(YTest, Ypred_dtree_test)
    dt_precision_test = metrics.precision_score(YTest, Ypred_dtree_test,
                                                average='binary')
    dtree_recall_test = metrics.recall_score(YTest, Ypred_dtree_test,
                                             average='binary')

    print "DT test prediction time: " + str(dtree_predict_time_test)
    print "DT test accuracy Score: " + str(dtree_accuracy_test)
    print "DT test precision Score: " + str(dt_precision_test)
    print "DT test recall Score: " + str(dtree_recall_test)

    print "Creating ROC curve"
    y_true = YTest
    y_score = tree_grid.predict_proba(XTest)
    fprSVM, trpSVM, _ = metrics.roc_curve(y_true=y_true,
                                          y_score=y_score[:, 0],
                                          pos_label=0)
    plt.plot(fprSVM, trpSVM, 'r-', label='DT')


def AdaBoost(X, Y, XTest, YTest):
    print '-----------------------------------------------------'

    # param_grid = {'learning_rate': [0.1, 0.3, 0.6, 1, 3, 6, 10]}

    # tree_grid = GridSearchCV(AdaBoostClassifier(), param_grid)
    tree_grid = AdaBoostClassifier(n_estimators=100, learning_rate=2)
    tree_grid.fit(X, Y)

    # print("The best parameters are %s with a score of %0.2f"
    #       % (tree_grid.best_params_, tree_grid.best_score_))

    print "Computing training statistics"
    dtree_predict_time_training = time.time()
    Ypred_dtree_training = tree_grid.predict(X)
    dtree_predict_time_training = time.time() - dtree_predict_time_training

    dtree_accuracy_training = metrics.accuracy_score(Y, Ypred_dtree_training)
    dt_precision_training = metrics.precision_score(Y, Ypred_dtree_training,
                                                    average='binary')
    dtree_recall_training = metrics.recall_score(Y, Ypred_dtree_training,
                                                 average='binary')

    print "DT training prediction time: " + str(dtree_predict_time_training)
    print "DT training accuracy Score: " + str(dtree_accuracy_training)
    print "DT training precision Score: " + str(dt_precision_training)
    print "DT training recall Score: " + str(dtree_recall_training)

    print "Computing testing statistics"
    dtree_predict_time_test = time.time()
    Ypred_dtree_test = tree_grid.predict(XTest)
    dtree_predict_time_test = time.time() - dtree_predict_time_test

    dtree_accuracy_test = metrics.accuracy_score(YTest, Ypred_dtree_test)
    dt_precision_test = metrics.precision_score(YTest, Ypred_dtree_test,
                                                average='binary')
    dtree_recall_test = metrics.recall_score(YTest, Ypred_dtree_test,
                                             average='binary')

    print "DT test prediction time: " + str(dtree_predict_time_test)
    print "DT test accuracy Score: " + str(dtree_accuracy_test)
    print "DT test precision Score: " + str(dt_precision_test)
    print "DT test recall Score: " + str(dtree_recall_test)

    print "Creating ROC curve"
    y_true = YTest
    y_score = tree_grid.predict_proba(XTest)
    fprSVM, trpSVM, _ = metrics.roc_curve(y_true=y_true,
                                          y_score=y_score[:, 0],
                                          pos_label=0)
    plt.plot(fprSVM, trpSVM, 'c-', label='ADA')


def BuildROCs():
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('ROC2.pdf')
