from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

clf = tree.DecisionTreeClassifier()
iris = load_iris()
dot_data = StringIO()
clf = clf.fit(iris.data, iris.target)
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
