import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from utils import euclidean, pairwise_euclidean

class KMeans:

    def __init__(self, k, samples):
        self.k = k
        self.samples = samples
        self.cluster_per_sample = np.random.randint(0, k, len(samples))

        # clusters are tuples (cluster_number, cluster_centroid)
        self.clusters = []

        # variances is an array of the average of W(Ck) for each epoch
        # on lesson the formula is not average (cumulative) but since
        # there can a cluster that disappears, we are changing it to avg.
        # TODO: ask about this
        self.variances = []

        for i in range(k):
            self.clusters.append((i, np.zeros(shape=self.samples.shape[1])))
        self.clusters = np.array(self.clusters, dtype=object)

    @staticmethod
    def get_sample_cluster(clusters, centroids):
        def f(sample):
            distance = euclidean(sample)
            min_distance = -1
            closest_cluster = -1
            for cluster, centroid in zip(clusters, centroids):
                d = distance(centroid)
                if min_distance == -1 or d < min_distance:
                    min_distance = d
                    closest_cluster = cluster
            return closest_cluster
        return f

    def run(self, epochs):
        for j in range(epochs):
            clusters = np.unique(self.cluster_per_sample)
            centroids = []
            # compute centroid for each cluster
            for c in clusters:
                cluster_elems_indexes = np.where(self.cluster_per_sample == c)[0]
                centroids.append(self.samples[cluster_elems_indexes].mean(axis=0))

            self.compute_variances(clusters)

            # compute new cluster for each observation
            current_clusters_per_sample = np.fromiter(
                map(self.get_sample_cluster(clusters, centroids), self.samples), dtype=int
            )
            if (self.cluster_per_sample == current_clusters_per_sample).all():
                print(f"Algorithm found a minimum at {j}.")
                return
            self.cluster_per_sample = current_clusters_per_sample

    def compute_variances(self, clusters):
        cluster_variances = []
        for c in clusters:
            cluster_elems_indexes = np.where(self.cluster_per_sample == c)[0]
            # TODO: should we consider distance = 0 if there is only one element inside the cluster ?
            distances = pairwise_euclidean(self.samples[cluster_elems_indexes])
            assert len(distances) > 0
            cluster_variances.append(distances.mean())

        self.variances.append(np.array(cluster_variances).mean())

    def plot_variances(self, filename):
        plt.plot(np.arange(len(self.variances)), self.variances)
        plt.title("Avg. variance of clusters")
        plt.xlabel("Epoch")
        plt.ylabel("Variance, C(W)")
        plt.savefig(f'./results/{filename}')
        plt.show()

    def __str__(self, complete=False):
        info = ''
        for cluster in self.clusters:
            info += f'Cluster {cluster[0]} with centroid {cluster[1]}\n'
        if complete:
            for (sample,cluster) in zip(self.samples, self.cluster_per_sample):
                info += f'Sample {sample} correspond to cluster {cluster}\n'
        return info


def code_method(max_k, samples, epochs, filename):
    variances = []
    for k in np.arange(1, max_k+1):
        k_means_model = KMeans(k, samples)
        k_means_model.run(epochs)
        # we append final variance (min)
        variances.append(k_means_model.variances[-1])
    plt.plot(np.arange(1, max_k+1), variances)
    plt.title("Avg. variance of clusters")
    plt.xlabel("K")
    plt.ylabel("Variance, C(W)")
    plt.savefig(f'./results/{filename}')
    plt.show()


# k_clusters = 5
data, y = make_blobs(n_samples=200, n_features=2, centers=10)
# k_means = KMeans(k_clusters, data)
# k_means.run(1000)
# k_means.plot_variances('variances_per_epoch.jpg')
code_method(40, data, 1000, 'code_method.jpg')



