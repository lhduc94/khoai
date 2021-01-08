from .base import SamplingMethod
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
import copy
import numpy as np


class GraphDensityAL(SamplingMethod):
    """https://www.mpi-inf.mpg.de/fileadmin/inf/d2/Research_projects_files/EbertCVPR2012.pdf"""

    def __init__(self, X, n_neighbors=10):
        super().__init__()
        self.X = X
        self.n_neighbors = n_neighbors
        self.gamma = 1. / (self.X.shape[1] * self.X.var())
        self.adjacency_matrix = None
        self.graph_density = None
        self.starting_density = None
        self.compute_graph_destiny()

    def update_gamma(self, gamma):
        self.gamma = gamma
        self.compute_graph_destiny()

    def compute_graph_destiny(self):

        adjacency_matrix = kneighbors_graph(self.X, self.n_neighbors, mode='connectivity', p=1)
        neighbors = adjacency_matrix.nonzero()
        inds = zip(neighbors[0], neighbors[1])

        for entry in inds:
            i = entry[0]
            j = entry[1]
            distance = pairwise_distances([self.X[i]], [self.X[j]], metric='manhattan')
            distance = distance[0, 0]
            weight = np.exp(-distance * self.gamma)
            adjacency_matrix[i, j] = weight
            adjacency_matrix[j, i] = weight
        self.adjacency_matrix = adjacency_matrix
        self.graph_density = np.zeros(self.X.shape[0])
        for i in np.arange(self.X.shape[0]):
            self.graph_density[i] = adjacency_matrix[i, :].sum() / (adjacency_matrix[i, :] > 0).sum()
        self.starting_density = copy.deepcopy(self.graph_density)

    def reverse(self):
        self.graph_density = copy.deepcopy(self.starting_density)

    def select_samples(self, N=None, already_selected=None, **kwargs):
        if N is None:
            N = self.X.shape[0]
        if already_selected is None:
            already_selected = []
        samples = list()
        self.graph_density[already_selected] = min(self.graph_density) - 1
        while len(samples) < N:
            selected = np.argmax(self.graph_density)
            neighbors = (self.adjacency_matrix[selected, :] > 0).nonzero()[1]
            self.graph_density[neighbors] = self.graph_density[neighbors] - self.graph_density[selected]
            samples.append(selected)
            self.graph_density[already_selected] = min(self.graph_density) - 1
            self.graph_density[samples] = min(self.graph_density) - 1
        return [], list(samples)
