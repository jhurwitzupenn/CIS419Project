import GraphAnalyzer as ga
import trainClassifiers

print "beginning script"
g = ga.GraphAnalyzer()
print "reading graphs"
g.ReadGraphs("./gmlFiles/")
print "building features"
g.BuildFeatures()
print "labeling features"
g.LabelFeatures()

X = g.features
Y = g.labels

print "standardizing features"
# Standardize features
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

trainClassifiers.SVMRBF(X, Y, X, Y)
trainClassifiers.SVMSig(X, Y, X, Y)
trainClassifiers.DTree(X, Y, X, Y)

# trainClassifiers.makePredictions(X, Y, svm_rbf, svm_sig, dtree)
