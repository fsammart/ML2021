from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from utils import get_confusion_matrix, get_precision
import numpy as np

COW = 0
GRASS = 1
SKY = 2


def get_class_data(filename, class_value):
    im_arr = np.asarray(Image.open(filename))
    samples = im_arr.reshape(im_arr.shape[0] * im_arr.shape[1], im_arr.shape[2])
    predictions = np.full(samples.shape[0], fill_value=class_value)
    return samples, predictions


cow_samples, cow_predictions = get_class_data('../data/vaca.jpg', COW)
grass_samples, grass_predictions = get_class_data('../data/pasto.jpg', GRASS)
sky_samples, sky_predictions = get_class_data('../data/cielo.jpg', SKY)

X = np.append(np.append(cow_samples, grass_samples, axis=0), sky_samples, axis=0)
y = np.append(np.append(cow_predictions, grass_predictions, axis=0), sky_predictions, axis=0)

assert cow_predictions.shape[0] + grass_predictions.shape[0] + sky_predictions.shape[0] == y.shape[0]

# For now we leave random state to reproduce results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def run_classifier(c_value, kernel='linear'):
    # multiclass support is handled by one-vs-one scheme
    classifier = svm.SVC(C=c_value, kernel=kernel)
    classifier.fit(X_train, y_train)

    test_predicted = classifier.predict(X_test)
    train_predicted = classifier.predict(X_train)

    assert len(test_predicted) == len(y_test)

    test_confusion = get_confusion_matrix(test_predicted, y_test)
    train_confusion = get_confusion_matrix(train_predicted, y_train)

    test_precision = get_precision(test_confusion)
    train_precision = get_precision(train_confusion)

    # correct = sum(1 for x in filter(lambda x: x, list(map(lambda x: x[0] == x[1], results))))
    # print(f'There are {correct} out of {len(X_test)} samples. This gives us an accuracy of {correct/len(X_test)}')

    print(f'Finished iteration with c {c_value} and kernel {kernel}')

    return test_precision, train_precision


c_range = np.arange(0.1, 5.2, 0.2)

filename = 'results.csv'
final_string = ''

for c in c_range:
    test_precision, train_precision = run_classifier(c)
    final_string += f'{c},linear,{test_precision},{train_precision}\n'

with open(filename, 'w') as fp:
    fp.write(final_string)
