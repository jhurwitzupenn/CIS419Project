import cPickle as pickle
import GraphAnalyzer as ga

print "beginning script"
g = ga.GraphAnalyzer()
print "reading graphs"
g.ReadGraphs("./gmlFiles/")

pickle.dump(g.graphs, open("../graphs.pkl", "wb"))
