import numpy as np
import math


def get_indices_of_k_smallest(arr, k):
    # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    idx = np.argpartition(arr.ravel(), k)
    return np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))].flatten()

def euclidean(v1):
    def f(v2):
        #return np.linalg.norm(v1-v2)
        result = np.sqrt(np.sum(np.square(v1 - v2)))
        assert math.isnan(result) is False
        return result
    return f


def pairwise_euclidean(elements):
    distances = []
    if len(elements) == 1:
        distances.append(0)
    else:
        for i in range(len(elements)):
            distance = euclidean(elements[i])
            for j in np.arange(i + 1, len(elements)):
                distances.append(distance(elements[j]))
    return np.array(distances)


def pairwise_euclidean_clusters(centroids, clusters):
    distances = []
    indexes = []

    # This case should not happen
    if len(centroids) == 1:
        distances.append(0)
        indexes.append(tuple([clusters[0], clusters[0]]))
    else:
        for i in range(len(centroids)):
            distance = euclidean(centroids[i])
            for j in np.arange(i + 1, len(centroids)):
                distances.append(distance(centroids[j]))
                indexes.append(tuple([clusters[i], clusters[j]]))
    return np.array(distances), indexes


def get_min_idx(array):
    if len(array) == 0:
        return
    min_value = array[0]
    index = 0
    for i in range(len(array)):
        if array[i] < min_value:
            min_value = array[i]
            index = i
    return index
