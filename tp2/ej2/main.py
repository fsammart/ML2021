import math
import numpy as np
import pandas as pd

from knn import KNN, WeightedKNN
from utils import confusion_matrix, plot_heatmap


target = ['star_rating']
attributes = ['wordcount', 'title_sentiment', 'sentiment_value']

sentiments = pd.read_csv('../data/reviews_sentiment.csv', delimiter=";")
sentiments = sentiments[target + attributes]

# preprocessing on title_sentiment column. TODO: decision with NaN must be reviewed
sentiments.loc[sentiments.title_sentiment == 'negative', 'title_sentiment'] = 0
sentiments.loc[sentiments.title_sentiment != 0, 'title_sentiment'] = 1

# TODO: I think they don't provide more information, discuss
# dropping duplicate values
sentiments = sentiments.drop_duplicates()
# shuffling elements before using ml algorithm
sentiments = sentiments.sample(frac=1).reset_index(drop=True)

print(f'After pre processing dataset we are left with {len(sentiments)} registers.\n')

# Item 1: answer the question below
print('Which is the avg qty of words of comments valued with 1 star?')
avg = round(
    sentiments.loc[sentiments.star_rating == 1]['wordcount'].mean(), 2
)
print(f'The avg. is {avg} words.\n')

classes = np.array(sentiments[target].star_rating.unique())
labels = np.array(sentiments[target].star_rating)
data = np.array(sentiments[attributes])  # each element is an array of 3 elements

crossed_validation_k = 4
chunk_size = math.floor(len(data)/crossed_validation_k)

print(f'For dataset with {len(data)} registers, each chunk is size {chunk_size}.', '\n')

for i in range(crossed_validation_k):

    # test indexes according to iteration
    test_indices = np.array(range(chunk_size * i, chunk_size * (i + 1), 1))

    # Item 2: split dataset into training and test
    X = np.delete(data, test_indices, axis=0)
    labels_X = np.delete(labels, test_indices, axis=0)
    Y = data[test_indices[0]:(test_indices[-1] + 1)]
    labels_Y = labels[test_indices[0]:(test_indices[-1] + 1)]

    knn = KNN(X, labels_X, classes)
    w_knn = WeightedKNN(X, labels_X, classes)

    results = knn.batch_predict(Y)
    w_results = w_knn.batch_predict(Y)

    knn_confusion = confusion_matrix(results, labels_Y, classes)
    w_knn_confusion = confusion_matrix(w_results, labels_Y, classes)

    plot_heatmap(knn_confusion, f'knn_5_i{i}.png')
    plot_heatmap(w_knn_confusion, f'w_knn_5_i{i}.png')

    print(f'Finished iteration {i}. Trained {len(X)} registers, tested {len(Y)} registers.')
