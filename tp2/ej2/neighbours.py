import math
import numpy as np
import pandas as pd

from knn import KNN, WeightedKNN
from utils import (
    confusion_matrix,
    normalize_dataframe,
    plot_to_choose_k
)


target = ['star_rating']
attributes = ['wordcount', 'title_sentiment', 'sentiment_value']

sentiments = pd.read_csv('../data/reviews_sentiment.csv', delimiter=";")
sentiments = sentiments[target + attributes]

# preprocessing on title_sentiment column.
sentiments['title_sentiment'].fillna(0.5, inplace=True)
sentiments['title_sentiment'].replace('negative', 0, inplace=True)
sentiments['title_sentiment'].replace('positive', 1, inplace=True)

# do not drop this because we loose information
# sentiments = sentiments.drop_duplicates()

knn_means = []
knn_stds = []
w_knn_means = []
w_knn_stds = []

neighbours = np.arange(start=2, stop=10, step=1)
print(f'Testing crossed validation for {neighbours} neighbours.')
for neigh_k in neighbours:

    iterations = np.arange(20)
    knn_precisions_avg = np.zeros(len(iterations))
    w_knn_precisions_avg = np.zeros(len(iterations))

    for it in iterations:

        # shuffling elements before using ml algorithm
        sentiments = sentiments.sample(frac=1).reset_index(drop=True)

        classes = np.array(sentiments[target].star_rating.unique())
        labels = np.array(sentiments[target].star_rating)
        data = np.array(normalize_dataframe(sentiments[attributes]))  # each element is an array of 3 elements

        crossed_validation_k = 5  #TODO: update with proper
        chunk_size = math.floor(len(data)/crossed_validation_k)

        # we will store here precision values for each cross validation run
        knn_precisions = np.zeros(crossed_validation_k)
        w_knn_precisions = np.zeros(crossed_validation_k)

        for i in range(crossed_validation_k):

            # test indexes according to iteration
            test_indices = np.array(range(chunk_size * i, chunk_size * (i + 1), 1))

            # Item 2: split dataset into training and test
            X = np.delete(data, test_indices, axis=0)
            labels_X = np.delete(labels, test_indices, axis=0)
            Y = data[test_indices[0]:(test_indices[-1] + 1)]
            labels_Y = labels[test_indices[0]:(test_indices[-1] + 1)]

            knn = KNN(X, labels_X, classes, k=neigh_k)
            w_knn = WeightedKNN(X, labels_X, classes, k=neigh_k)

            results = knn.batch_predict(Y)
            w_results = w_knn.batch_predict(Y)

            knn_confusion = confusion_matrix(results, labels_Y, classes)
            w_knn_confusion = confusion_matrix(w_results, labels_Y, classes)

            knn_tp = knn_confusion.trace()
            w_knn_tp = w_knn_confusion.trace()

            knn_precisions[i] = knn_tp/knn_confusion.sum()
            w_knn_precisions[i] = w_knn_tp/w_knn_confusion.sum()

        knn_precisions_avg[it] = knn_precisions.mean()
        w_knn_precisions_avg[it] = w_knn_precisions.mean()

    knn_means.append(knn_precisions_avg.mean())
    knn_stds.append(knn_precisions_avg.std())
    w_knn_means.append(w_knn_precisions_avg.mean())
    w_knn_stds.append(w_knn_precisions_avg.std())

plot_to_choose_k(
    knn_means,
    knn_stds,
    w_knn_means,
    w_knn_stds,
    title='Precisión promedio vs. elección de K vecinos.',
    filename='k_neighbours/choose_k.png'
)

print(f'Completed {len(neighbours)} experiments with crossed validation.')