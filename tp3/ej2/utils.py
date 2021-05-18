import numpy as np
import matplotlib.pyplot as plt

COW = 0
GRASS = 1
SKY = 2

COW_COLOR = (219, 149, 156)
GRASS_COLOR = (61, 130, 69)
SKY_COLOR = (145, 215, 235)


def get_confusion_matrix(predictions, truths):
    matrix = np.zeros(shape=(3, 3))
    for pred, truth in zip(predictions, truths):
        matrix[truth][pred] += 1
    return matrix


def get_precision(confusion_matrix):
    sum_columns = np.sum(confusion_matrix, axis=0)
    diagonal = np.diagonal(confusion_matrix)

    return np.mean(diagonal/sum_columns)

def plot_c_results():
    # format of data is c_value, kernel=linear, test_precision, train_precision
    data = np.genfromtxt('c_results.csv', delimiter=',')
    c_values = []
    test_precisions = []
    train_precisions = []
    for d in data:
        c_values.append(round(float(d[0]), 2))
        test_precisions.append(float(d[2]))
        train_precisions.append(float(d[3]))

    plt.plot(test_precisions, color='magenta', label='test')
    plt.plot(train_precisions, color='royalblue', label='train')
    plt.xlabel("Valor C")
    plt.ylabel("Precisión")
    plt.title("Precisión SVM vs. C")
    plt.legend()
    plt.xticks(ticks=np.arange(len(c_values)), labels=c_values)

    plt.savefig("c_value_test_train.png")


def plot_kernel_results():
    # format of data is c_value, kernel=linear, test_precision, train_precision
    data = np.genfromtxt('kernel_results.csv', delimiter=',')
    kernels = ['linear', 'poly', 'rbf']
    test_precisions = []
    train_precisions = []
    for d in data:
        test_precisions.append(float(d[2]))
        train_precisions.append(float(d[3]))

    plt.plot(test_precisions, color='magenta', label='test')
    plt.plot(train_precisions, color='royalblue', label='train')
    plt.xlabel("Kernel")
    plt.ylabel("Precisión")
    plt.title("Precisión SVM vs. Kernel")
    plt.legend()
    plt.xticks(ticks=np.arange(len(kernels)), labels=kernels)

    plt.savefig("kernel_test_train.png")


def plot_heatmap(matrix, filename):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)

    for i, j in np.ndindex(matrix.shape):
        c = matrix[i][j]
        ax.text(i, j, str(c), va='center', ha='center')

    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.savefig(f'{filename}', bbox_inches='tight')
    plt.close()


def map_color(x):
    if x == COW:
        return COW_COLOR
    if x == GRASS:
        return GRASS_COLOR
    if x == SKY:
        return SKY_COLOR
