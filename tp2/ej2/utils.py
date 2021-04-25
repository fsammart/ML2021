import numpy as np
import matplotlib.pyplot as plt

results_directory = './results'


def normalize_dataframe(data):
    result = data.copy()
    for feature_name in data.columns:
        max_value = data[feature_name].max()
        min_value = data[feature_name].min()
        result[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)
    return result


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
