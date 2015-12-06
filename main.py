import trainClassifiers
import cPickle as pickle

X = pickle.load(open("features.p", "rb"))
Y = pickle.load(open("labels.p", "rb"))

XTest = pickle.load(open("featuresTest.p", "rb"))
YTest = pickle.load(open("labelsTest.p", "rb"))

print "Standardizing features"
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# modify labels so 2 becomes 1 and 1 becomes 0
Y = Y - 1
YTest = YTest - 1

print "Standardizing featuresTest"
XTest = (XTest - mean) / std

trainClassifiers.SVM(X, Y, XTest, YTest)
# # trainClassifiers.SVMSig(X, Y, X, Y)
trainClassifiers.DTree(X, Y, XTest, YTest)
trainClassifiers.AdaBoost(X, Y, XTest, YTest)
