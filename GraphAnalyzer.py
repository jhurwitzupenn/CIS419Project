import networkx
import glob
import os


class GraphAnalyzer(object):
    """Class that reads and analyzes GML encoded graphs"""
    def __init__(self):
        super(GraphAnalyzer, self).__init__()
        self.graphs = {}

    def ReadGraphs(self, pathToGraphs):
        os.chdir(pathToGraphs)
        for file in glob.glob('*.gml'):
            self.graphs[os.path.splitext(file)[0]] = networkx.read_gml(file)
