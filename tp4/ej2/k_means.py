import numpy as np
from sklearn.datasets import make_blobs
from utils import euclidean


class KMeans:

    def __init__(self, k, samples):
        self.k = k
        self.samples = samples
        self.cluster_per_sample = np.random.randint(0, k, len(samples))
        self.centroids = np.zeros(shape=(k, self.samples.shape[1]))

    def get_sample_cluster(self, sample):
        distances = np.fromiter(map(euclidean(sample), self.centroids), dtype=np.float64)
        return np.where(distances == np.amin(distances))[0]

    def run(self, epochs):
        print(self.samples)
        print(self.cluster_per_sample)
        for j in range(epochs):
            # compute centroid for each cluster
            for i in range(self.k):
                # TODO: clusters can be empty, what then?
                cluster_elems_indexes = np.where(self.cluster_per_sample == i)[0]
                centroid = self.samples[cluster_elems_indexes].mean(axis=0)
                self.centroids[i] = centroid
            print(f'{j}: {self.centroids}')

            # compute new cluster for each observation
            clusters = np.fromiter(map(self.get_sample_cluster, self.samples), dtype=int)
            if (self.cluster_per_sample == clusters).all():
                print(f"Algorithm found a minimum at {j}.")
                return
            self.cluster_per_sample = clusters


k = 3
data, y = make_blobs(n_samples=24, n_features=3, centers=3)
k_means = KMeans(k, data)
k_means.run(1)
