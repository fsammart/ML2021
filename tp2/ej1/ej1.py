import csv
import random
import tp2.ej1.DecisionTree as dt
import numpy as np
from tp2.ej1.RandomForest import RandomForest
from tp2.ej1.utils import initialize_confusion_matrix, add_to_confusion_matrix
import pandas as pd
from tp2.ej2.utils import plot_heatmap
from collections import defaultdict

DURATION_MONTH = "Duration of Credit (month)"
CREDIT_AMOUNT = "Credit Amount"
AGE = "Age (years)"

gain_function = dt.DecisionTree.gini
height_limit = 2
number_of_elements = 100


# Column Names: Creditability ,Account Balance,
# Duration of Credit (month),Payment Status of Previous Credit,
# Purpose,Credit Amount,Value Savings/Stocks,Length of current employment,
# Instalment per cent,Sex & Marital Status,Guarantors,Duration in Current address,
# Most valuable available asset,Age (years),Concurrent Credits,Type of apartment,
# No of Credits at this Bank,Occupation,No of dependents,Telephone,Foreign Worker

def discretize_month (value) :
    if int(value) <= 12 :
        return "short"
    elif int(value) <= 25 :
        return "medium"
    else:
        return "long"


def discretize_credit_amount (value) :
    if int(value) <= 1000 :
        return "small"
    elif int(value) <= 3000 :
        return "medium"
    else:
        return "big"


def discretize_age (value) :
    if int(value) <= 30:
        return "young"
    elif int(value) <= 60:
        return "adult"
    else:
        return "old"


def discretize (row) :
    row[DURATION_MONTH] = discretize_month(row[DURATION_MONTH])
    row[CREDIT_AMOUNT] = discretize_credit_amount(row[CREDIT_AMOUNT])
    row[AGE] = discretize_age(row[AGE])
    return row


# Returns Training and Test samples already discretized
def get_credit_prepared_data (train_percentage) :
    with open('tp2/data/german_credit.csv') as f :
        data = list(csv.DictReader(f))
    # Jump first row
    process = data[1:]
    for observation in process:
        discretize(observation)

    random.shuffle(data)

    training_samples = int(len(data) * train_percentage)
    training_data = data[:training_samples]
    prediction_data = data[training_samples :]
    attributes = data[0]
    return training_data, prediction_data, attributes



def run_decision_tree (gain_function, height_limit=None, gain_umbral=None, data_umbral_explode=None):
    result_variable = "Creditability"
    training_data, prediction_data, attributes = get_credit_prepared_data(0.8)

    decision_tree = dt.DecisionTree(training_data,
                                    attributes,
                                    result_variable,
                                    "credit",
                                    gain_function,
                                    gain_umbral=gain_umbral,
                                    data_umbral_explode=data_umbral_explode,
                                    height_limit=height_limit)

    # decision_tree.create_dot_image()

    train_matrix = defaultdict(lambda: defaultdict(int))
    initialize_confusion_matrix(train_matrix, ['1', '0'])


    #Confusion matrix for training data
    for observation in training_data :
        predicted_value = decision_tree.predict(observation)
        add_to_confusion_matrix(train_matrix, observation[result_variable], predicted_value)

        
    test_matrix = defaultdict(lambda: defaultdict(int))
    initialize_confusion_matrix(test_matrix, ['1', '0'])

    for observation in prediction_data:
        predicted_value = decision_tree.predict(observation)
        add_to_confusion_matrix(test_matrix, observation[result_variable], predicted_value)

    # print ("Expanded nodes: {}\n".format(decision_tree.expanded_nodes))
    # print(pd.DataFrame.from_dict(matrix))
    return decision_tree, train_matrix, test_matrix


def run_random_forest(gain_function, height_limit=None, gain_umbral=None, data_umbral_explode=None):
    result_variable = "Creditability"
    training_data, prediction_data, attributes = get_credit_prepared_data(0.8)
    random_f = RandomForest(training_data, attributes, result_variable, "credit_forest",gain_function, 
                            6,len(attributes)-1,int(len(training_data)/2),height_limit=height_limit)
    
    train_matrix = defaultdict(lambda: defaultdict(int))
    initialize_confusion_matrix(train_matrix, ['1', '0'])

    #Confusion matrix for training data
    for observation in training_data :
        predicted_value = random_f.predict(observation)
        add_to_confusion_matrix(train_matrix, observation[result_variable], predicted_value)

    test_matrix = defaultdict(lambda: defaultdict(int))
    initialize_confusion_matrix(test_matrix, ['1', '0'])
    for observation in prediction_data:
        predicted_value = random_f.predict(observation)
        add_to_confusion_matrix(test_matrix, observation[result_variable], predicted_value)
    
    # random_f.create_dot_image()

    return random_f, train_matrix, test_matrix


def get_metrics(matrix):
    tp = matrix['1']['1']
    tn = matrix['0']['0']
    fn = matrix['1']['0']
    fp = matrix['0']['1']
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp+fp)
    recall = tp / (tp + fn)
    f1 = (2*precision*recall) / (precision + recall)
    tvp = tp /  (tp+fn)
    tfp = fp / (fp + tn)

    # Devuelvo precision pero se pueden devolver otras
    return precision

def run_single_dt():
    data_umbral = 50
    gain_umbral = None
    filename = "confusion_data_50.png"
    height = None
    dtree, train_matrix, test_matrix =run_decision_tree(dt.DecisionTree.shannon, height_limit=height, data_umbral_explode=data_umbral, gain_umbral=gain_umbral)
    print(train_matrix)
    cm = conf_matrix(test_matrix)
    plot_heatmap(cm, filename)

def conf_matrix(output_matrix):

    print(pd.DataFrame.from_dict(output_matrix))
    matrix = np.zeros((2,2))
    for i in output_matrix.keys():
        for j in output_matrix[i].keys():
            matrix[int(i)][int(j)] = output_matrix[i][j]
    
    return matrix


def run_random_forest_experiment():
    number_of_repetitions = 100
    data_umbral = None
    gain_umbral = None
    filename = "random_forest_precision.csv"
    height = None
    for height in range(1,15):
        for i in range(number_of_repetitions):
            random_f, train_matrix, test_matrix =run_random_forest(dt.DecisionTree.shannon, height_limit=height, data_umbral_explode=data_umbral, gain_umbral=gain_umbral)
            train_precision = get_metrics(train_matrix)
            test_precision = get_metrics(test_matrix)
            with open(filename, "a") as f:
                f.write("{},{},{},{}, {}, {}\n".format(height,random_f.expanded_nodes, data_umbral, gain_umbral, train_precision, test_precision))

def run_single_random_forest():

    data_umbral = 30
    gain_umbral = None
    filename = "random_forest_data_30.png"
    height = None
    random_f, train_matrix, test_matrix = run_random_forest(gain_function, height_limit=height, gain_umbral=gain_umbral, data_umbral_explode=data_umbral)
    cm = conf_matrix(test_matrix)
    plot_heatmap(cm, filename)

def run_dt_experiment():
    min_value = 0.03
    max_value = 0.08
    number_of_repetitions = 50
    data_umbral = None
    gain_umbral = None
    filename = "gain _umbral.csv"
    height = None
    rng = np.linspace(min_value, max_value, 500)
    for gain_umbral in rng:
        for i in range(number_of_repetitions):
            dtree, train_matrix, test_matrix =run_decision_tree(dt.DecisionTree.shannon, height_limit=height, data_umbral_explode=data_umbral, gain_umbral=gain_umbral)
            train_precision = get_metrics(train_matrix)
            test_precision = get_metrics(test_matrix)
            with open(filename, "a") as f:
                f.write("{},{},{},{}, {}, {}\n".format(height,dtree.expanded_nodes, data_umbral, gain_umbral, train_precision, test_precision))

def main ( ) :
    
    run_random_forest_experiment()
    

    # initialize_confusion_matrix(matrix, ['1', '0'])

    # for observation in prediction_data :
    #     predicted_value = decision_tree.predict(observation)
    #     add_to_confusion_matrix(matrix, observation[result_variable], predicted_value)

    # print(pd.DataFrame.from_dict(matrix))

    ## Now Random Forest
    

main()