import trainClassifiers
import cPickle as pickle

X = pickle.load(open("features.p", "rb"))
Y = pickle.load(open("labels.p", "rb"))

print "standardizing features"
# Standardize features
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

print len(X)
print len(Y)

trainClassifiers.SVM(X, Y, X, Y)
# trainClassifiers.SVMSig(X, Y, X, Y)
trainClassifiers.DTree(X, Y, X, Y)
