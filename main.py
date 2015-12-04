import GraphAnalyzer as ga
import trainClassifiers

g = ga.GraphAnalyzer()
g.ReadGraphs("./gmlFiles/")
g.BuildFeatures()
g.LabelFeatures()

X = g.features
Y = g.labels

# Standardize features
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

trainClassifiers.SVMRBF(X, Y, X, Y)
trainClassifiers.SVMSig(X, Y, X, Y)
trainClassifiers.DTree(X, Y, X, Y)

# trainClassifiers.makePredictions(X, Y, svm_rbf, svm_sig, dtree)
