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

    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.savefig(f'{results_directory}/confusion/{filename}', bbox_inches='tight')
    plt.close()


def plot_precision(precisions_knn, precisions_w_knn, k, filename):

    plt.title(f'Precisión por iteración, k={k}')
    plt.ylabel('Precisión')
    plt.xlabel('Iteración')

    points = np.arange(len(precisions_knn))
    width = 0.35  # the width of the bars

    plt.bar(points - width/2, precisions_knn, width, label='KNN', color='paleturquoise')
    plt.bar(points + width/2, precisions_w_knn, width, label='Weighted KNN', color='plum')

    plt.xticks(points)
    plt.legend(loc='lower left')

    plt.savefig(f'{results_directory}/precision/{filename}', bbox_inches='tight')
    plt.close()


def plot_different_k(precisions_knn, precisions_w_knn, filename, title):

    plt.title(title)
    plt.ylabel('Precisión')
    plt.xlabel('K')

    plt.plot(precisions_knn, label='KNN', color='paleturquoise')
    plt.plot(precisions_w_knn, label='Weighted KNN', color='plum')

    points = np.arange(len(precisions_knn))
    labels = np.arange(start=2, stop=len(precisions_knn)+2, step=1)
    plt.xticks(points, labels)
    plt.legend()

    plt.savefig(f'{results_directory}/{filename}', bbox_inches='tight')
    plt.close()
