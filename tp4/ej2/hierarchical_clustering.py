import numpy as np
import sys
from tp4.ej2.utils import (
    pairwise_euclidean_clusters,
    get_min_idx,
    get_sample_cluster,
    euclidean,
    get_two_closest
)

class HierarchicalClustering:

    # Data are samples
    # k is number of clusters to stop.
    # e.g if k=2, at the end there will be
    # only two clusters left
    def __init__(self, samples, k):
        self.samples = samples
        self.k = k
        self.cluster_per_sample = np.arange(0, len(self.samples))

        # this will be initialized when calling add_cluster_classification
        self.cluster_classifications = None
        self.clusters = None
        self.centroids = None
        self.distances = np.zeros((len(samples), len(samples)))
        # initialize distance matrix
        for idx1, cords1 in enumerate(samples) :
            for idx2, cords2 in enumerate(samples) :
                self.distances[idx1][idx2] = euclidean(cords1)(cords2)
        # diagonal with inf values.
        np.fill_diagonal(self.distances, sys.maxsize)

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
            print(f'Dealing with {len(clusters)} clusters ({self.k} goal).')
            centroids = self.get_centroids(self.cluster_per_sample)
            _, _ = get_two_closest(centroids, self.distances, self.cluster_per_sample, self.samples)
            print(f'Clusters per sample > {self.cluster_per_sample}')
            # distances, indexes = pairwise_euclidean_clusters(centroids, clusters)
            # (c1, c2) = indexes[get_min_idx(distances)]
            # c2_elems_indexes = np.where(self.cluster_per_sample == c2)[0]
            # for i in c2_elems_indexes:
            #     self.cluster_per_sample[i] = c1

    # this should be called only when finishing run
    def add_cluster_classification(self, labels):
        cluster_classifications = []
        clusters = np.unique(self.cluster_per_sample)
        for c in clusters:
            cluster_elems_indexes = np.where(self.cluster_per_sample == c)[0]
            cluster_labels = labels[cluster_elems_indexes]
            print(f'Counts for cluster {c} are {np.bincount(cluster_labels[:,0])}')
            winner = np.bincount(cluster_labels[:,0]).argmax()
            cluster_classifications.append(winner)

        self.cluster_classifications = cluster_classifications
        self.clusters = clusters.tolist()
        self.centroids = self.get_centroids(clusters)

    def predict(self, samples):
        predictor = get_sample_cluster(self.clusters, self.centroids)
        winner_clusters = list(map(predictor, samples))
        predictions = list(map(lambda c: self.cluster_classifications[self.clusters.index(c)], winner_clusters))
        return predictions
