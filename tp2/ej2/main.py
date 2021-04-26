import math
import numpy as np
import pandas as pd

from knn import KNN, WeightedKNN
from utils import (
    confusion_matrix,
    normalize_dataframe,
    plot_heatmap,
    plot_precision,
    plot_different_k
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

# shuffling elements before using ml algorithm
sentiments = sentiments.sample(frac=1).reset_index(drop=True)

print(f'After pre processing dataset we are left with {len(sentiments)} registers.\n')

# Item 1: answer the question below
print('Which is the avg qty of words of comments valued with 1 star?')
avg = round(
    sentiments.loc[sentiments.star_rating == 1]['wordcount'].mean(), 2
)
print(f'The avg. is {avg} words.\n')
print('Which is the avg qty of words of comments valued with 2 stars?')
avg = round(
    sentiments.loc[sentiments.star_rating == 2]['wordcount'].mean(), 2
)
print(f'The avg. is {avg} words.\n')
print('Which is the avg qty of words of comments valued with 3 stars?')
avg = round(
    sentiments.loc[sentiments.star_rating == 3]['wordcount'].mean(), 2
)
print(f'The avg. is {avg} words.\n')
print('Which is the avg qty of words of comments valued with 4 stars?')
avg = round(
    sentiments.loc[sentiments.star_rating == 4]['wordcount'].mean(), 2
)
print(f'The avg. is {avg} words.\n')
print('Which is the avg qty of words of comments valued with 5 stars?')
avg = round(
    sentiments.loc[sentiments.star_rating == 5]['wordcount'].mean(), 2
)
print(f'The avg. is {avg} words.\n')

classes = np.array(sentiments[target].star_rating.unique())
labels = np.array(sentiments[target].star_rating)
data = np.array(normalize_dataframe(sentiments[attributes])) # each element is an array of 3 elements

crossed_validation_k = 6
chunk_size = math.floor(len(data)/crossed_validation_k)

print(f'For dataset with {len(data)} registers, each chunk is size {chunk_size}.', '\n')

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

    # knn = KNN(X, labels_X, classes)
    w_knn = WeightedKNN(X, labels_X, classes)

    # results = knn.batch_predict(Y)
    w_results = w_knn.batch_predict(Y)

    # knn_confusion = confusion_matrix(results, labels_Y, classes)
    w_knn_confusion = confusion_matrix(w_results, labels_Y, classes)

    # knn_tp = knn_confusion.trace()
    w_knn_tp = w_knn_confusion.trace()

    # knn_precisions[i] = knn_tp/knn_confusion.sum()
    w_knn_precisions[i] = w_knn_tp/w_knn_confusion.sum()

    # plot_heatmap(knn_confusion, f'knn_k_{crossed_validation_k}_i{i}.png')
    plot_heatmap(w_knn_confusion, f'w_knn_k_{crossed_validation_k}_i{i}.png')

plot_precision(knn_precisions, w_knn_precisions, crossed_validation_k, filename=f'precision_k_{crossed_validation_k}.png')

print()
print(f'Finished all iterations. Trained {len(data)} registers, tested {chunk_size} registers.')
print(f'For knn, avg precision was {knn_precisions.mean()} with max value {knn_precisions.max()}')
print(f'For weighted knn, avg precision was {w_knn_precisions.mean()} with max value {w_knn_precisions.max()}')
