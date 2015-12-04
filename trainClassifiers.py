from sklearn import svm, tree
from sklearn import metrics
import time
from sklearn.externals.six import StringIO
import pydot

"""
=====================================
Train Shortest Path Classifiers
=====================================

Author: Reuben Abrahamn, Nivedita Sankar, Jordan Hurwitz

Adapted from scikit_learn documentation.

"""
print(__doc__)


def trainClassifiers(X, Y):
    C = 0.01
    gam = 0.01
    # grid search over these to find parameters
    svm_rbf_model = svm.SVC(C=C, kernel='rbf', gamma=gam)
    svm_sig_model = svm.SVC(C=C, kernel='sigmoid', gamma=gam)
    dtree_model = tree.DecisionTreeClassifier(max_depth=3)

    # fit the models
    print "Training the SVM RBF model..."
    rbf_train_time = time.time()
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

    dot_data = StringIO()
    tree.export_graphviz(dtree_model, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("../dtree.pdf")

    print "SVM RBF Training time: " + str(rbf_train_time)
    print "SVM Sigmoid Training time: " + str(sig_train_time)
    print "DecisionTree Training time: " + str(dtree_train_time)

    return svm_rbf_model, svm_sig_model, dtree_model


def SVMRBF(X, Y, Xtest, Ytest):
    C = 0.01
    gam = 0.01
    # grid search over these to find parameters
    svm_rbf_model = svm.SVC(C=C, kernel='rbf', gamma=gam)

    # fit the models
    print "Training the SVM RBF model..."
    rbf_train_time = time.time()
    svm_rbf_model.fit(X, Y)
    rbf_train_time = time.time() - rbf_train_time

    print "SVM RBF Training time: " + str(rbf_train_time)

    print "Predicting with the SVM RBF model..."
    rbf_predict_time = time.time()
    Ypred_rbf = svm_rbf_model.predict(X)
    rbf_predict_time = time.time() - rbf_predict_time

    rbf_accuracy = metrics.accuracy_score(Ytest, Ypred_rbf)
    rbf_precision = metrics.precision_score(Ytest, Ypred_rbf, average='macro')
    rbf_recall = metrics.recall_score(Ytest, Ypred_rbf, average='macro')

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
    sig_precision = metrics.precision_score(Ytest, Ypred_sig, average='macro')
    sig_recall = metrics.recall_score(Ytest, Ypred_sig, average='macro')

    print "SVM Sigmoid Prediction time: " + str(sig_predict_time)
    print "SVM Sigmoid Accuracy Score: " + str(sig_accuracy)
    print "SVM Sigmoid Precision Score: " + str(sig_precision)
    print "SVM Sigmoid Recall Score: " + str(sig_recall)


def DTree(X, Y, Xtest, Ytest):
    dtree_model = tree.DecisionTreeClassifier(max_depth=3)

    print "Training the DecisionTree model..."
    dtree_train_time = time.time()
    dtree_model.fit(X, Y)
    dtree_train_time = time.time() - dtree_train_time

    dot_data = StringIO()
    tree.export_graphviz(dtree_model, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("../dtree.pdf")

    print "DecisionTree Training time: " + str(dtree_train_time)

    print "Predicting with the Decision Tree model..."
    dtree_predict_time = time.time()
    Ypred_dtree = dtree_model.predict(X)
    dtree_predict_time = time.time() - dtree_predict_time

    dtree_accuracy = metrics.accuracy_score(Ytest, Ypred_dtree)
    dt_precision = metrics.precision_score(Ytest, Ypred_dtree, average='macro')
    dtree_recall = metrics.recall_score(Ytest, Ypred_dtree, average='macro')

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
