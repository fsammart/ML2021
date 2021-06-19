import numpy as np


def get_indices_of_k_smallest(arr, k):
    # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    idx = np.argpartition(arr.ravel(), k)
    return np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))].flatten()

def euclidean(v1):
    def f(v2):
        return np.linalg.norm(v1-v2)
    return f


def pairwise_euclidean(elements, clusters):
    distances = []
    indexes = []
    for i in range(len(elements)):
        distance = euclidean(elements[i])
        for j in np.arange(i + 1, len(elements)):
            distances.append(distance(elements[j]))
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
