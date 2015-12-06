import networkx
import gmlReader
import glob
import os
import numpy as np
import time
import gc


class IncrementalGraphAnalyzer(object):
    """Class that reads and analyzes GML encoded graphs"""
    def __init__(self):
        super(IncrementalGraphAnalyzer, self).__init__()
        self.features = []
        self.labels = []

    def BuildFeatures(self, graph):
        graphFeatures = []
        graphFeatures.append(self._ComputeMaxDegreeCentrality(graph))
        graphFeatures.append(self._ComputeMinDegreeCentrality(graph))
        graphFeatures.append(self._ComputeAvgClustering(graph))
        graphFeatures.append(self._ComputeEdgeNodeRatio(graph))
        graphFeatures.append(self._ComputeMaxEdgeLength(graph))
        graphFeatures.append(self._ComputeDensity(graph))
        graphFeatures.append(self._ComputeCorrelationCoefficient(graph))
        return graphFeatures

    def LabelFeature(self, graph):
        # for each graph
        # pick a random source and a random target
        # run each of the networkx src tgt shortest path algorithms one by one
        # time how long they each take
        # repeat for N different srcs/tgts
        # find the average time for each algorithm
        # make the label for that graph the one with the shortest time
        # feature key: 0 = dijkstra, 1 = bidijkstra 2 = astar
        numIters = 10
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

        minTime = min(meanDijkstra, meanBiDijkstra, meanAStar)
        if meanDijkstra == minTime:
            label = 0
        elif meanBiDijkstra == minTime:
            label = 1
        else:
            label = 2

        return label

    def BuildAllFeatures(self, pathToGraphs):
        os.chdir(pathToGraphs)
        files = glob.glob('*.gml')
        numFiles = len(files)
        for file in files:
            print numFiles
            numFiles -= 1
            print 'Reading file: {0}'.format(file)
            g = gmlReader.read_gml(file)
            print 'Building features for file: {0}'.format(file)
            self.features.append(self.BuildFeatures(g))
            print 'Building labels for file: {0}'.format(file)
            self.labels.append(self.LabelFeature(g))
            gc.collect()
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)

    def _ComputeMaxDegreeCentrality(self, graph):
        degrees = networkx.degree_centrality(graph)
        return max(degrees.iteritems())[1]

    def _ComputeMinDegreeCentrality(self, graph):
        degrees = networkx.degree_centrality(graph)
        return min(degrees.iteritems())[1]

    def _ComputeAvgClustering(self, graph):
        return networkx.average_clustering(graph)

    def _ComputeEdgeNodeRatio(self, graph):
        return float(len(graph.edges())) / len(graph.nodes())

    def _ComputeMaxEdgeLength(self, graph):
        maxLength = 0
        for node in graph:
            for neighbor in graph[node]:
                edgeLength = graph[node][neighbor]['length']
                if maxLength < edgeLength:
                    maxLength = edgeLength
        return maxLength

    def _ComputeRadius(self, graph):
        return networkx.radius(graph)

    def _ComputeDensity(self, graph):
        return networkx.density(graph)

    def _ComputeCorrelationCoefficient(self, graph):
        return networkx.degree_pearson_correlation_coefficient(graph)
