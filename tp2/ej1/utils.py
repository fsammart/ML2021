import itertools
from collections import defaultdict


def initialize_confusion_matrix(matrix, classes):
    for p in itertools.product(classes, repeat=2):
        if matrix.get(p[0], None) is None:
            matrix[p[0]] = dict()
        matrix[p[0]][p[1]] = 0


def add_to_confusion_matrix(matrix, category, predicted_value):
    matrix[category][predicted_value] += 1

