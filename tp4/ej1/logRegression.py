import csv
from collections import defaultdict
from tp4.utils.utils import *
from sklearn.linear_model import LogisticRegression
import pandas as pd


def read_data(attrs, selected_attr, target_name):
    with open("../data/acath.csv") as f:
        data = list(csv.DictReader(f, delimiter=';'))
        for row in data:
            for attr in attrs:
                if attr not in selected_attr and attr != target_name:
                    row.pop(attr)

    return data


def get_attr_or_moda(elem, attr, freq_dict):
    if elem != "":
        return float(elem)
    return float(max(freq_dict[attr], key=freq_dict[attr].get))


def fill_empty(data):
    # Fill empty values with more frequent value of attribute.
    freq_dict = defaultdict(lambda: defaultdict(int))
    for observation in data:
        for attr in observation:
            if observation[attr] != "":
                freq_dict[attr][observation[attr]] += 1

    for x in data:
        for attr in x:
            # Get attribute or if empty moda
            x[attr] = get_attr_or_moda(x[attr], attr, freq_dict)

    return data



def separate_target_variable(data, attributes, class_name):
    data_labels = np.stack([int(x[class_name]) for x in data])

    data_samples = []
    for x in data:
        row = []
        for attr in attributes:
            if attr == class_name:
                continue
            row.append(x[attr])
        data_samples.append(row)
    data_samples = np.array(data_samples)

    return data_samples, data_labels


def lr_train(trainx, trainy):
    # all params set to default, maybe check what means each of them
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(trainx, trainy)
    return lr


def lr_predict(lr, test_x):
    predictions = lr.predict(test_x)
    return predictions


def confusion_matrix(predictions, test_y):
    return get_confusion_matrix(predictions, test_y)


def logistic_regression(attributes, selected_attributes, class_name):
    data = read_data(attributes, selected_attributes, class_name)
    data = fill_empty(data)
    # We divide training ad testing
    train_p = 0.7
    training_amount = int(train_p * len(data))
    train, test = divide_data(data, training_amount)


    train_x, train_y = separate_target_variable(train, selected_attributes, class_name)
    test_x, test_y = separate_target_variable(test, selected_attributes, class_name)

    lr_model = lr_train(train_x, train_y)
    predictions = lr_predict(lr_model, test_x)

    # Use score method to get accuracy of model
    score = lr_model.score(test_x, test_y)
    print("-" * 20)
    print(f'Variables: {selected_attributes}')
    print(f'Accuracy: {round(score, 3)}')

    matrix_ = confusion_matrix(predictions, test_y )

    print(f'Confusion matrix')
    print(pd.DataFrame.from_dict(matrix_))
    print("-" * 20)

    print(f'Precision')
    print(get_precision(matrix_))
    print("-" * 20)
    print(f'Coefficients')
    print(lr_model.coef_, lr_model.intercept_)

    return lr_model, train, test



class_name = "sigdz"
attributes = ['sex', 'age', 'cad.dur', 'choleste', 'tvdlm']
selected_attributes = ['age', 'cad.dur', 'choleste']
print("#" * 20)
print("Logistic Regression")
print("#" * 20)

lr, training_data, test_data = logistic_regression(attributes, selected_attributes, class_name)
prob_list = [70, 1, 150]  # age, duration, choleste
print(f'Probabilities for {prob_list}')
probabilities = lr.predict_proba(np.array(prob_list).reshape(-1, 1).T)
print(f"Probability sigdz = 1, {round(probabilities[0][1], 3)}")
print(f"Probability sigdz = 0, {round(probabilities[0][0], 3)}")