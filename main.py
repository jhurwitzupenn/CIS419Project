import GraphAnalyzer as ga
import trainClassifiers

g = ga.GraphAnalyzer()
g.ReadGraphs("./gmlFiles/")
g.BuildFeatures()
g.LabelFeatures()

X = g.features
Y = g.labels

trainClassifiers.SVMRBF(X, Y, X, Y)
trainClassifiers.SVMSig(X, Y)
trainClassifiers.DTree(X, Y)

# trainClassifiers.makePredictions(X, Y, svm_rbf, svm_sig, dtree)
