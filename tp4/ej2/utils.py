import numpy as np


def get_indices_of_k_smallest(arr, k):
    # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    idx = np.argpartition(arr.ravel(), k)
    return np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))].flatten()

def euclidean(v1):
    def f(v2):
        return np.linalg.norm(v1-v2)
    return f
