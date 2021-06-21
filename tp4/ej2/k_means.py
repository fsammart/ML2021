import numpy as np
import matplotlib.pyplot as plt
from utils import pairwise_euclidean, get_sample_cluster

class KMeans:

    def __init__(self, k, samples):
        self.k = k
        self.samples = samples
        self.cluster_per_sample = np.random.randint(0, k, len(samples))

        # variances is an array of the average of W(Ck) for each epoch
        # on lesson the formula is not average (cumulative) but since
        # there can a cluster that disappears, we are changing it to avg.
        # TODO: ask about this
        self.variances = []

        # this will be initialized when calling add_cluster_classification
        self.cluster_classifications = None
        self.clusters = None
        self.centroids = None

    def get_centroids(self, clusters):
        centroids = []
        for c in clusters:
            cluster_elems_indexes = np.where(self.cluster_per_sample == c)[0]
            centroid = self.samples[cluster_elems_indexes].mean(axis=0)
            centroids.append(centroid)
        return centroids

    def run(self, epochs):
        for j in range(epochs):
            clusters = np.unique(self.cluster_per_sample)
            # compute centroid for each cluster
            centroids = self.get_centroids(clusters)

            self.compute_variances(clusters)

            # compute new cluster for each observation
            current_clusters_per_sample = np.fromiter(
                map(get_sample_cluster(clusters, centroids), self.samples), dtype=int
            )
            if (self.cluster_per_sample == current_clusters_per_sample).all():
                return
            self.cluster_per_sample = current_clusters_per_sample

    # this should be called only when finishing run
    def add_cluster_classification(self, labels):
        cluster_classifications = []
        clusters = np.unique(self.cluster_per_sample)
        for c in clusters:
            cluster_elems_indexes = np.where(self.cluster_per_sample == c)[0]
            cluster_labels = labels[cluster_elems_indexes]
            print(f'Counts for cluster {c} are {np.bincount(cluster_labels[:, 0])}')
            winner = np.bincount(cluster_labels[:, 0]).argmax()
            cluster_classifications.append(winner)

        self.cluster_classifications = cluster_classifications
        self.clusters = clusters.tolist()
        self.centroids = self.get_centroids(clusters)

    def predict(self, samples):
        predictor = get_sample_cluster(self.clusters, self.centroids)
        winner_clusters = list(map(predictor, samples))
        predictions = list(map(lambda c: self.cluster_classifications[self.clusters.index(c)], winner_clusters))
        return predictions

    def compute_variances(self, clusters):
        cluster_variances = []
        for c in clusters:
            cluster_elems_indexes = np.where(self.cluster_per_sample == c)[0]
            # TODO: should we consider distance = 0 if there is only one element inside the cluster ?
            distances = pairwise_euclidean(self.samples[cluster_elems_indexes])
            cluster_variances.append(distances.mean())

        self.variances.append(np.array(cluster_variances).mean())
        # TODO: check why this is weird
        #self.variances.append(np.array(cluster_variances).sum())

    def plot_variances(self, filename):
        plt.plot(np.arange(len(self.variances)), self.variances)
        plt.title("Avg. variance of clusters")
        plt.xlabel("Epoch")
        plt.ylabel("Variance, C(W)")
        plt.savefig(f'./results/{filename}')
        plt.show()


def code_method(max_k, samples, epochs, filename, repeat=10):
    averages = []
    stds = []

    for k in np.arange(1, max_k+1):
        k_variances = []
        for i in range(repeat):
            k_means_model = KMeans(k, samples)
            k_means_model.run(epochs)
            # we append final variance (min)
            k_variances.append(k_means_model.variances[-1])
            print(f'Remaining clusters {len(np.unique(k_means_model.cluster_per_sample))} out of {k}')
        averages.append(np.mean(k_variances))
        stds.append(np.std(k_variances))
        print(f'Finished with k={k}')

    plt.errorbar(
        np.arange(1, max_k+1), averages, stds,
        linestyle='-', label='K', capsize=5, marker='o'
    )
    plt.title("Avg. variance of clusters")
    plt.xlabel("K")
    plt.ylabel("Variance, C(W)")
    plt.savefig(f'./results/{filename}')
    plt.show()
