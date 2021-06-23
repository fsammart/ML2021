import numpy as np
import math
import sys
import matplotlib.pyplot as plt


def get_indices_of_k_smallest (arr, k) :
    # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    idx = np.argpartition(arr.ravel(), k)
    return np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))].flatten()


def euclidean (v1) :
    def f (v2) :
        # return np.linalg.norm(v1-v2)
        result = np.sqrt(np.sum(np.square(v1 - v2)))
        assert math.isnan(result) is False
        return result

    return f


def pairwise_euclidean (elements) :
    distances = []
    if len(elements) == 1 :
        distances.append(0)
    else :
        for i in range(len(elements)) :
            distance = euclidean(elements[i])
            for j in np.arange(i + 1, len(elements)) :
                distances.append(distance(elements[j]))
    return np.array(distances)

def get_centroid_of_cluster(cluster, cluster_per_sample, samples):
        cluster_elems_indexes = np.where(cluster_per_sample == cluster)[0]
        centroid = samples[cluster_elems_indexes].mean(axis=0)
        return centroid

def centroid (col_index, row_index, matrix_distances, centroids, indexes, indexes_col):
    d = euclidean(centroids[col_index])

    for i in range(0, matrix_distances.shape[0]) :
        if i != col_index and i != row_index and i not in indexes  and i not in indexes_col :
            dist_centroid = d(centroids[i])
            matrix_distances[col_index][i] = dist_centroid
            matrix_distances[i][col_index] = dist_centroid

def get_centroids( clusters, cluster_per_sample, samples):
        centr= []
        for c in clusters:
            cluster_elems_indexes = np.where(cluster_per_sample == c)[0]
            centroid = samples[cluster_elems_indexes].mean(axis=0)
            centr.append(centroid)
        return centr

def get_two_closest (centroids, distances, cluster_per_sample, samples) :
    indx = np.argmin(distances)

    col_index = int(indx / distances.shape[0])
    row_index = indx % distances.shape[1]


    indexes = np.where(cluster_per_sample == cluster_per_sample[row_index])[0]
    indexes_col = np.where(cluster_per_sample == cluster_per_sample[col_index])[0]

    for elem in indexes :
        cluster_per_sample[elem] = cluster_per_sample[col_index]
        for i in range(0, distances.shape[0]) :
            distances[elem][i] = sys.maxsize
            distances[i][elem] = sys.maxsize

    for elem in indexes_col:
        if elem is not col_index:
            for i in range(0, distances.shape[0]) :
                distances[elem][i] = sys.maxsize
                distances[i][elem] = sys.maxsize

    centroids = get_centroids(cluster_per_sample, cluster_per_sample, samples)

    centroid(col_index, row_index, distances, centroids, indexes, indexes_col)



    return col_index, row_index


def pairwise_euclidean_clusters (centroids, clusters) :
    distances = []
    indexes = []

    # This case should not happen
    if len(centroids) == 1 :
        distances.append(0)
        indexes.append(tuple([clusters[0], clusters[0]]))
    else :
        for i in range(len(centroids)) :
            distance = euclidean(centroids[i])
            for j in np.arange(i + 1, len(centroids)) :
                distances.append(distance(centroids[j]))
                indexes.append(tuple([clusters[i], clusters[j]]))
    return np.array(distances), indexes


def get_min_idx (array) :
    if len(array) == 0 :
        return
    min_value = array[0]
    index = 0
    for i in range(len(array)) :
        if array[i] < min_value :
            min_value = array[i]
            index = i
    return index


def get_sample_cluster (clusters, centroids) :
    def f (sample) :
        distance = euclidean(sample)
        min_distance = -1
        closest_cluster = -1
        for cluster, centroid in zip(clusters, centroids) :
            d = distance(centroid)
            if min_distance == -1 or d < min_distance :
                min_distance = d
                closest_cluster = cluster
        return closest_cluster

    return f


def get_confusion_matrix(predictions, truths):
    matrix = np.zeros(shape=(2, 2))
    for pred, truth in zip(np.array(predictions), np.array(truths)):
        matrix[truth][pred] += 1
    return matrix


def get_accuracy(confusion_matrix):
    trues = np.sum(np.diag(confusion_matrix))
    total = confusion_matrix.sum()
    return trues/total

def plot_heatmap(matrix, filename, v_min=0, v_max=1052):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues, vmin=v_min, vmax=v_max)

    for i, j in np.ndindex(matrix.shape):
        c = matrix[i][j]
        ax.text(j, i, str(c), va='center', ha='center')

    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.savefig(f'{filename}', bbox_inches='tight')
    plt.close()