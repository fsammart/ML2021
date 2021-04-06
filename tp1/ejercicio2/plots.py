import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Even though the name says matrix, this is a dictionary that
# has, for every key (which is the real category),
# another dictionary as the value with categories as names
# and qty of test news predicted with that category as values
def plot_confusion_matrix(confusion_matrix: dict):
    matrix = []
    for _, result in confusion_matrix.items():
        matrix.append(list(result.values()))
    matrix = np.array(matrix)

    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)

    categories = list(confusion_matrix.keys())

    # Adds number to heatmap matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            c = matrix[i][j]
            ax.text(i, j, str(c), va='center', ha='center')

    # Replace labels
    x_labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(len(categories)):
        x_labels[i+1] = categories[i]
    ax.set_xticklabels(x_labels)

    y_labels = [item.get_text() for item in ax.get_yticklabels()]
    for i in range(len(categories)):
        y_labels[i+1] = categories[i]
    ax.set_yticklabels(y_labels)

    plt.xlabel("Predicción")
    plt.ylabel("Categoría Real")
    plt.savefig('confussion_same_probability.png', bbox_inches='tight')

    # Shows image
    plt.show()


def plot_evaluation_metrics():
    columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    metrics = pd.read_csv('/home/marina/ML2021/tp1/data/Metrics.csv')

    appearances = metrics.loc[metrics['Method'] == 1]
    quantity = metrics.loc[metrics['Method'] == 2]

    appearances_means = []
    quantity_means = []
    appearances_stds = []
    quantity_stds = []

    for c in columns:
        appearances_values = np.array(appearances[c])
        quantity_values = np.array(quantity[c])

        appearances_means.append(np.mean(appearances_values))
        appearances_stds.append(np.std(appearances_values))

        quantity_means.append(np.mean(quantity_values))
        quantity_stds.append(np.std(quantity_values))

    plt.xlabel("Métrica de evaluación")
    plt.ylabel("Valor")
    plt.errorbar(
        columns,
        appearances_means,
        appearances_stds,
        linestyle='None',
        label='Method Appearances',
        capsize=5,
        marker='o'
    )
    plt.errorbar(
        columns,
        quantity_means,
        quantity_stds,
        linestyle='None',
        label='Method Quantities',
        capsize=5,
        marker='o'
    )
    plt.legend()
    plt.savefig('avg.png')
    plt.show()


#plot_evaluation_metrics()