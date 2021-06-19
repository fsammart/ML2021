import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_blobs
from utils import pairwise_euclidean, get_min_idx

class HierarchicalClustering:

    # Data are samples
    # k is number of clusters to stop.
    # e.g if k=2, at the end there will be
    # only two clusters left
    def __init__(self, samples, k):
        self.samples = samples
        self.k = k
        self.cluster_per_sample = np.arange(0, len(self.samples))


    def get_centroids(self, clusters):
        centroids = []
        for c in clusters:
            cluster_elems_indexes = np.where(self.cluster_per_sample == c)[0]
            centroid = self.samples[cluster_elems_indexes].mean(axis=0)
            centroids.append(centroid)
        return centroids

    def run(self):
        while self.k < len(np.unique(self.cluster_per_sample)):
            clusters = np.unique(self.cluster_per_sample)
            centroids = self.get_centroids(clusters)
            distances, indexes = pairwise_euclidean(centroids, clusters)
            (c1, c2) = indexes[get_min_idx(distances)]
            c2_elems_indexes = np.where(self.cluster_per_sample == c2)[0]
            for i in c2_elems_indexes:
                self.cluster_per_sample[i] = c1


data, y = make_blobs(n_samples=4, n_features=1, centers=2)
print(data)
hc = HierarchicalClustering(data, k=2)
hc.run()
