import numpy as np
import matplotlib.pyplot as plt

results_directory = './results'


def confusion_matrix(predictions, reals, classes):
    matrix = np.zeros((len(classes), len(classes)))

    for pred, truth in zip(predictions, reals):
        matrix[int(truth-1)][int(pred-1)] += 1

    return matrix


def plot_heatmap(matrix, filename):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)

    for i, j in np.ndindex(matrix.shape):
        c = matrix[i][j]
        ax.text(i, j, str(c), va='center', ha='center')

    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    plt.savefig(f'{results_directory}/{filename}', bbox_inches='tight')
    plt.clf()
