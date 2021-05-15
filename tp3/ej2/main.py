from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
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

# multiclass support is handled by one-vs-one scheme
classifier = svm.SVC()
classifier.fit(X_train, y_train)
y_predicted = classifier.predict(X_test)

assert len(y_predicted) == len(y_test)

results = zip(list(y_predicted), list(y_test))

correct = sum(1 for x in filter(lambda x: x, list(map(lambda x: x[0] == x[1], results))))
print(f'There are {correct} out of {len(X_test)} samples. This gives us an accuracy of {correct/len(X_test)}')







