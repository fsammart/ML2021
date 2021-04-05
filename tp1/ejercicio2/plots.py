import numpy as np
import matplotlib.pyplot as plt


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

    # Shows image
    plt.show()
