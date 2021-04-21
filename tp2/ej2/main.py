import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

print(f'After pre processing dataset we are left with {len(sentiments)} registers.\n')

# Item 1: answer the question below
print('Which is the avg qty of words of comments valued with 1 star?')
avg = round(
    sentiments.loc[sentiments.star_rating == 1]['wordcount'].mean(), 2
)
print(f'The avg. is {avg} words.\n')

# TODO: update this
crossed_validation_k = 3
for i in range(crossed_validation_k):

    classes = np.array(sentiments[target].star_rating.unique())
    labels = np.array(sentiments[target].star_rating)
    data = np.array(sentiments[attributes])  # each element is an array of 3 elements

    # Item 2: split dataset into training and test
    X, Y, labels_X, labels_Y = train_test_split(data, labels, test_size=0.2, shuffle=True)

    knn = KNN(X, labels_X, classes)
    w_knn = WeightedKNN(X, labels_X, classes)

    results = knn.batch_predict(Y)
    w_results = w_knn.batch_predict(Y)

    knn_confusion = confusion_matrix(results, labels_Y, classes)
    w_knn_confusion = confusion_matrix(w_results, labels_Y, classes)

    plot_heatmap(knn_confusion, f'knn_5_i{i}.png')
    plot_heatmap(w_knn_confusion, f'w_knn_5_i{i}.png')

    print(f'Finished iteration {i}. Run batch prediction for {len(Y)} registers.\n')
