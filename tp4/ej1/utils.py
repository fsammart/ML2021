import numpy as np
import matplotlib.pyplot as plt
import random

COW = 0
GRASS = 1
SKY = 2

COW_COLOR = (219, 149, 156)
GRASS_COLOR = (61, 130, 69)
SKY_COLOR = (145, 215, 235)

def divide_data(data, training_size):
    random.shuffle(data)
    training_data = data[:training_size]
    test_data = data[training_size:]
    return training_data, test_data


def get_confusion_matrix(predictions, truths):
    matrix = np.zeros(shape=(2, 2))
    print(len(predictions[0]))
    print(len(truths["sigdz"]))
    for pred, truth in zip(predictions[0], truths["sigdz"]):
        matrix[truth][pred] += 1
    return matrix


def get_accuracy(confusion_matrix):
    trues = np.sum(np.diag(confusion_matrix))
    total = confusion_matrix.sum()
    return trues / total

def get_precision(confusion_matrix):
    # print(confusion_matrix)
    sum_columns = np.sum(confusion_matrix, axis=0)
    diagonal = np.diagonal(confusion_matrix)
    # print(diagonal)
    # print(sum_columns) 
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

