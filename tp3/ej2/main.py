from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from utils import get_confusion_matrix, get_precision, plot_heatmap, map_color
import numpy as np
import cv2

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
    plot_heatmap(test_confusion, f'../confusions/test_c_{c_value}_kernel_{kernel}.png')
    plot_heatmap(train_confusion, f'../confusions/train_c_{c_value}_kernel_{kernel}.png')

    test_precision = get_precision(test_confusion)
    train_precision = get_precision(train_confusion)

    # correct = sum(1 for x in filter(lambda x: x, list(map(lambda x: x[0] == x[1], results))))
    # print(f'There are {correct} out of {len(X_test)} samples. This gives us an accuracy of {correct/len(X_test)}')

    print(f'Finished iteration with c {c_value} and kernel {kernel}')

    return test_precision, train_precision


def run_c_values():
    c_range = np.arange(20)

    filename = 'c_results_bis.csv'
    final_string = ''

    for c in c_range:
        test_precision, train_precision = run_classifier(c)
        final_string += f'{c},linear,{test_precision},{train_precision}\n'


def run_kernels():
    kernels = ['linear', 'poly', 'rbf']

    filename = 'kernel_results.csv'
    final_string = ''

    for ker in kernels:
        test_precision, train_precision = run_classifier(c_value=1.0, kernel=ker)
        final_string += f'{1},{ker},{test_precision},{train_precision}\n'

    with open(filename, 'w') as fp:
        fp.write(final_string)


def run_cow(image, c_value=1.0, kernel='linear'):
    # multiclass support is handled by one-vs-one scheme
    classifier = svm.SVC(C=c_value, kernel=kernel)
    classifier.fit(X_train, y_train)

    samples, _ = get_class_data(image, 4)
    original = np.asarray(Image.open(image))

    predicted = classifier.predict(samples)
    result_image = np.array(list(map(map_color, predicted))).reshape(original.shape)

    output = np.hstack([original, result_image])
    im = Image.fromarray(output.astype(np.uint8))
    im.save('cow_vs_predicted.png')


run_cow('../data/cow.jpg')
