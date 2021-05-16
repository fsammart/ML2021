import numpy as np

COW = 0
GRASS = 1
SKY = 2


def get_confusion_matrix(predictions, truths):
    matrix = np.zeros(shape=(3, 3))
    for pred, truth in zip(predictions, truths):
        matrix[truth][pred] += 1
    return matrix


def get_precision(confusion_matrix):
    sum_columns = np.sum(confusion_matrix, axis=0)
    diagonal = np.diagonal(confusion_matrix)

    return np.mean(diagonal/sum_columns)
