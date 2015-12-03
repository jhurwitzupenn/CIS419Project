import networkx
import gmlReader
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
        self._ComputeAvgClustering(graphs)
        self._ComputeEdgeNodeRatio(graphs)
        self._ComputeMaxEdgeLength(graphs)

        # Build X
        features = []
        for key in self.featuresByGraph.keys():
            features.append(self.featuresByGraph[key])
        self.features = numpy.array(features)

    def ReadGraphs(self, pathToGraphs):
        os.chdir(pathToGraphs)
        for file in glob.glob('*.gml'):
            graphName = os.path.splitext(file)[0]
            self.graphs[graphName] = gmlReader.read_gml(file)
            self.featuresByGraph[graphName] = []

    def _ComputeMaxDegreeCentrality(self, graphs):
        print 'Computing Max Degree Centrality'
        for key in graphs.keys():
            degrees = networkx.degree_centrality(graphs[key])
            self.featuresByGraph[key].append(max(degrees.iteritems())[1])

    def _ComputeMinDegreeCentrality(self, graphs):
        print 'Computing Min Degree Centrality'
        for key in graphs.keys():
            degrees = networkx.degree_centrality(graphs[key])
            self.featuresByGraph[key].append(min(degrees.iteritems())[1])

    def _ComputeMaxLoadCentrality(self, graphs):
        print 'Computing Max Load Centrality'
        for key in graphs.keys():
            degrees = networkx.load_centrality(graphs[key])
            self.featuresByGraph[key].append(max(degrees.iteritems())[1])

    def _ComputeMinLoadCentrality(self, graphs):
        print 'Computing Min Load Centrality'
        for key in graphs.keys():
            degrees = networkx.load_centrality(graphs[key])
            self.featuresByGraph[key].append(min(degrees.iteritems())[1])

    def _ComputeAvgClustering(self, graphs):
        print 'Computing Avg Clustering'
        for key in graphs.keys():
            self.featuresByGraph[key].append(
                networkx.average_clustering(graphs[key]))

    def _ComputeEdgeNodeRatio(self, graphs):
        print 'Computing edge to node Ratio'
        for key in graphs.keys():
            graph = graphs[key]
            self.featuresByGraph[key].append(
                float(len(graph.edges())) / len(graph.nodes()))

    def _ComputeMaxEdgeLength(self, graphs):
        print 'Computing Max Edge Length'
        maxLength = 0
        for graph in graphs:
            for node in graphs[graph]:
                for neighbor in graphs[graph][node]:
                    edgeLength = graphs[graph][node][neighbor]['length']
                    if maxLength < edgeLength:
                        maxLength = edgeLength
            self.featuresByGraph[graph].append(maxLength)

    def _ComputeRadius(self, graphs):
        print 'Computing radius'
        for key in graphs.keys():
            self.featuresByGraph[key].append(networkx.radius(graphs[key]))
