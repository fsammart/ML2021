import csv
import random
import tp2.ej1.DecisionTree as dt
from tp2.ej1.RandomForest import RandomForest
from tp2.ej1.utils import *
import pandas as pd

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
    if int(value) <= 6 :
        return "short"
    elif int(value) <= 12 :
        return "medium"
    elif int(value) <= 18 :
        return "long"
    else :
        return "very long"


def discretize_credit_amount (value) :
    if int(value) <= 1000 :
        return "small"
    elif int(value) <= 3000 :
        return "medium"
    elif int(value) <= 6000 :
        return "big"
    else :
        return "very big"


def discretize_age (value) :
    if int(value) <= 30 :
        return "young"
    elif int(value) <= 60 :
        return "adult"
    elif int(value) <= 80 :
        return "old"
    else :
        return "very old"


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


def main ( ) :
    result_variable = "Creditability"
    training_data, prediction_data, attributes = get_credit_prepared_data(0.8)

    decision_tree = dt.DecisionTree(training_data, attributes, result_variable, "credit", gain_function, height_limit)

    decision_tree.create_dot_image()
    initialize_confusion_matrix(matrix, ['1', '0'])

    for observation in prediction_data :
        predicted_value = decision_tree.predict(observation)
        add_to_confusion_matrix(matrix, observation[result_variable], predicted_value)

    print(pd.DataFrame.from_dict(matrix))

    ## Now Random Forest
    #data, attributes, target_variable, name, gain_function, number_of_trees,
    #number_of_attributes, number_of_elements,
    # height_limit = None) :
    random_f = RandomForest(training_data, attributes, result_variable, "credit_forest",gain_function,
                            6,len(attributes)-1,int(len(training_data)/2),7)

    initialize_confusion_matrix(matrix, ['1', '0'])

    for observation in prediction_data :
        predicted_value = random_f.predict(observation)
        add_to_confusion_matrix(matrix, observation[result_variable], predicted_value)

    print(pd.DataFrame.from_dict(matrix))
    random_f.create_dot_image()

main()