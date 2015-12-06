import networkx
import gmlReader
import glob
import os
import numpy as np
import time


class GraphAnalyzer(object):
    """Class that reads and analyzes GML encoded graphs"""
    def __init__(self):
        super(GraphAnalyzer, self).__init__()
        self.graphs = {}
        self.featuresByGraph = {}
        self.features = None
        self.labels = None

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
        self.features = np.array(features)

    def LabelFeatures(self):
        # for each graph
        # pick a random source and a random target
        # run each of the networkx src tgt shortest path algorithms one by one
        # time how long they each take
        # repeat for N different srcs/tgts
        # find the average time for each algorithm
        # make the label for that graph the one with the shortest time
        # feature key: 0 = dijkstra, 1 = bidijkstra 2 = astar
        n, d = self.features.shape
        labels = np.zeros(n)
        graphs = self.graphs
        numIters = 10
        _ = time.time()
        count = 0

        for graphName in graphs:
            graph = graphs[graphName]
            n = networkx.number_of_nodes(graph)
            dijkstraTimes = np.zeros(numIters)
            biDijkstraTimes = np.zeros(numIters)
            aStarTimes = np.zeros(numIters)
            for i in xrange(numIters):
                # pick a random source and target
                src = np.random.randint(0, n) + 1
                tgt = np.random.randint(0, n) + 1
                while tgt == src:
                    tgt = np.random.randint(0, n) + 1

                dijkstraTime = time.time()
                try:
                    networkx.dijkstra_path(graph, src, tgt)
                except:
                    # no path found
                    i -= 1
                    continue

                dijkstraTime = time.time() - dijkstraTime
                dijkstraTimes[i] = dijkstraTime

                biDijkstraTime = time.time()
                networkx.bidirectional_dijkstra(graph, src, tgt)
                biDijkstraTime = time.time() - biDijkstraTime
                biDijkstraTimes[i] = biDijkstraTime

                aStarTime = time.time()
                networkx.astar_path(graph, src, tgt)
                aStarTime = time.time() - aStarTime
                aStarTimes[i] = aStarTime

            meanDijkstra = np.mean(dijkstraTimes)
            meanBiDijkstra = np.mean(biDijkstraTimes)
            meanAStar = np.mean(aStarTimes)

            label = 0
            if meanDijkstra < meanBiDijkstra and meanDijkstra < meanAStar:
                label = 0
            elif meanBiDijkstra < meanDijkstra and meanBiDijkstra < meanAStar:
                label = 1
            else:
                label = 2
            labels[count] = label
            count += 1

        self.labels = labels

    def ReadGraphs(self, pathToGraphs):
        os.chdir(pathToGraphs)
        for file in glob.glob('*.gml'):
            print 'Reading file: {0}'.format(file)
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
