import networkx
import glob
import os
import numpy


class GraphAnalyzer(object):
    """Class that reads and analyzes GML encoded graphs"""
    def __init__(self):
        super(GraphAnalyzer, self).__init__()
        self.graphs = {}
        self.featuresByGraph = {}
        self.features = None

    def BuildFeatures(self):
        graphs = self.graphs
        self._ComputeMaxDegreeCentrality(graphs)
        self._ComputeMinDegreeCentrality(graphs)
        self._ComputeMaxLoadCentrality(graphs)
        self._ComputeMinLoadCentrality(graphs)
        self._ComputeAvgClustering(graphs)
        self._ComputeNodeEdgeRatio(graphs)
        self._ComputeRadius(graphs)

        # Build X
        features = []
        for key in self.featuresByGraph.keys():
            features.append(self.featuresByGraph[key])
        self.features = numpy.array(features)

    def ReadGraphs(self, pathToGraphs):
        os.chdir(pathToGraphs)
        for file in glob.glob('*.gml'):
            graphName = os.path.splitext(file)[0]
            self.graphs[graphName] = networkx.read_gml(file)
            self.featuresByGraph[graphName] = []

    def _ComputeMaxDegreeCentrality(self, graphs):
        for key in graphs.keys():
            degrees = networkx.degree_centrality(graphs[key])
            self.featuresByGraph[key].append(max(degrees.iteritems())[1])

    def _ComputeMinDegreeCentrality(self, graphs):
        for key in graphs.keys():
            degrees = networkx.degree_centrality(graphs[key])
            self.featuresByGraph[key].append(min(degrees.iteritems())[1])

    def _ComputeMaxLoadCentrality(self, graphs):
        for key in graphs.keys():
            degrees = networkx.load_centrality(graphs[key])
            self.featuresByGraph[key].append(max(degrees.iteritems())[1])

    def _ComputeMinLoadCentrality(self, graphs):
        for key in graphs.keys():
            degrees = networkx.load_centrality(graphs[key])
            self.featuresByGraph[key].append(min(degrees.iteritems())[1])

    def _ComputeAvgClustering(self, graphs):
        for key in graphs.keys():
            self.featuresByGraph[key].append(
                networkx.average_clustering(graphs[key]))

    def _ComputeNodeEdgeRatio(self, graphs):
        for key in graphs.keys():
            graph = graphs[key]
            self.featuresByGraph[key].append(
                float(len(graph.nodes())) / len(graph.edges()))

    def _ComputeRadius(self, graphs):
        for key in graphs.keys():
            self.featuresByGraph[key].append(networkx.radius(graphs[key]))
