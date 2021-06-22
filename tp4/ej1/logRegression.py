import csv
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import statsmodels.api as sm

from utils import *
from tp4.ej2.preprocessing import load_data, replace_nan, scale_data


def read_data(attrs, selected_attr, target_name):
    with open("../data/acath.csv") as f:
        data = list(csv.DictReader(f, delimiter=';'))
        for row in data:
            for attr in attrs:
                if attr not in selected_attr and attr != target_name:
                    row.pop(attr)

    return data

def balance_data(data, target_feature):
    random.shuffle(data)
    target_count = {}
    for x in data:
        v = x[target_feature]
        if v in target_count:
            target_count[v] +=1
        else:
            target_count[v] = 1

    min_value = min(target_count.values())
    for i in target_count:
        target_count[i] -= min_value

    for x in data:
        v = x[target_feature]
        if target_count[v] > 0:
            data.remove(x)
            target_count[v] -= 1
    
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


def logistic_regression(
        attributes,
        balance=None,
        remove_duplicates=False
):
    # data = read_data(attributes, selected_attributes, class_name)
    #
    # # We can try to balance the data before filling it with mean
    # #data = balance_data(data, class_name)
    # data = fill_empty(data)
    #
    # # We divide training ad testing
    # train_p = 0.7
    # training_amount = int(train_p * len(data))
    # train, test = divide_data(data, training_amount)
    # train_x, train_y = separate_target_variable(train, selected_attributes, class_name)
    # test_x, test_y = separate_target_variable(test, selected_attributes, class_name)

    label = ['sigdz']
    dataframe, dataframe_info = load_data('../data/acath.csv', attributes + label, balance=balance)
    if remove_duplicates:
        with_removed = dataframe.drop_duplicates(subset=attributes)
        print(f'Removed {len(dataframe)-len(with_removed)} duplicates from dataframe.')
        dataframe = with_removed

    replacement_info = replace_nan(dataframe)
    print('Nan values per column in original dataframe')
    print(replacement_info)

    data = dataframe[attributes]
    labels = dataframe[label]
    # no need since library does it
    #std_scaler, data = scale_data(data)

    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.3, random_state=42)

    lr_model = lr_train(train_x, train_y)
    predictions = lr_predict(lr_model, test_x)

    pdt_x = pd.DataFrame(train_x)
    pdt_x.columns = selected_attributes

    # TODO: check how to handle categorical features
    if "sex" in selected_attributes:
        pdt_x["sex"] = pd.Categorical(pdt_x["sex"])
        # pdt_x = pd.get_dummies(pdt_x)
        print(pdt_x)

    # Adding constant to train_x to add the Intercept term
    # https://stats.stackexchange.com/questions/203740/logistic-regression-scikit-learn-vs-statsmodels
    pdt_x = sm.add_constant(pdt_x)
    pdt_y = pd.DataFrame(train_y)

    logit_model=sm.Logit(pdt_y,pdt_x.astype(float))

    result=logit_model.fit(maxiter=100)
    print(result.summary())

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

    # antes decia train y test, se estaban usando??
    return lr_model, None, None



class_name = "sigdz"
attributes = ['sex', 'age', 'cad.dur', 'choleste', 'tvdlm']
selected_attributes = ['sex', 'age', 'cad.dur', 'choleste']
print("#" * 20)
print("Logistic Regression")
print("#" * 20)

lr, training_data, test_data = logistic_regression(selected_attributes, remove_duplicates=True)

prob_list = [1, 70, 1, 150]  # age, duration, choleste
print(f'Probabilities for {prob_list}')

#TODO: Aplicar formula de logistic
probabilities = lr.predict_proba(np.array(prob_list).reshape(-1, 1).T)
print(f"Probability sigdz = 1, {round(probabilities[0][1], 3)}")
print(f"Probability sigdz = 0, {round(probabilities[0][0], 3)}")