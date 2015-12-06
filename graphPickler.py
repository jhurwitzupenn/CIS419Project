import cPickle as pickle
import GraphAnalyzer as ga
import IncrementalGraphAnalyzer as iga

# print "beginning script"
# g = ga.GraphAnalyzer()
# print "reading graphs"
# g.ReadGraphs("./gmlFiles/")
# print "building features"
# g.BuildFeatures()
# print "labeling features"
# g.LabelFeatures()

print "beginning script"
g = iga.IncrementalGraphAnalyzer()
print "Reading graphs and building features"
g.BuildAllFeatures("./gmlFiles/")

features = g.features
labels = g.labels

pickle.dump(features, open("../features.p", "wb"))
pickle.dump(labels, open("../labels.p", "wb"))
