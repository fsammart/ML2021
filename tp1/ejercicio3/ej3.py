import csv
import itertools
from collections import defaultdict


# graph definition. Each line is oriented top-bottom
#
#           RANK
#         /   |   \
#        GRE  |  GPA
#         \   |   /
#           ADMIT


g = {"rank": [],
     "admit": ["GRE", "GPA", "rank"],
     "GRE": ["rank"],
     "GPA": ["rank"]
}

# Used for total probability theorem and to calculate laplacian correction
attribute_values = {"admit": [True, False], "GRE": [">=500", "<500"], "GPA": [">=3", "<3"],
                    "rank": [1, 2, 3, 4]}

conditional_probabilities = []
prior_probabilities = []


# extract takes values of indexes columns.
def extract(item, indexes):
    return tuple(item[i] for i in indexes)


def count_ocurrences(attributes, combination, data, positions, sums):
    indexes = []
    for attribute in attributes:
        indexes.append(positions[attribute])
    for item in data:
        if combination not in sums:
            sums[combination] = 0
        if extract(item, indexes) == combination:
            sums[combination] += 1
    return sums

def discretize(col, attr):
    if attr == "GRE" and col >= 500:
        return ">=500"
    elif attr == "GRE" and col <= 500:
        return "<500"
    elif attr == "GPA" and col >= 3:
        return ">=3"
    elif attr == "GPA" and col < 3:
        return "<3"
    elif attr == "admit":
        return bool(int(col) == 1)
    elif attr == "rank" :
        return int(col)


def prior_probability(position_of_attribute, data, prior_probabilities, values_for_attribute):
    # here we have to calculate relative frequency.
    freq = defaultdict(float)
    for item in data:
        freq[item[position_of_attribute]] += 1

    total = sum(freq.values())

    for k in freq:
        freq[k] = (freq[k] + 1) / (total + len(values_for_attribute))

    prior_probabilities.append(freq)

def train():
    with open("binary.csv") as f :
        reader = csv.reader(f, delimiter=',')
        next(reader)
        data = [(discretize(int(col2), "admit"), discretize(int(col2), "GRE"),
                 discretize(float(col3), "GPA"), discretize(col4,"rank")) for
                col1, col2, col3, col4 in reader]

        # We store column position of each attribute for easy access.
        positions = {"admit" : 0, "GRE" : 1, "GPA" : 2, "rank" : 3}

        for k, v in g.items() :
            if not v :
                # It has no father, then we must calculate prior probabilities.
                prior_probability(positions[k], data, prior_probabilities, attribute_values[k])
            else :
                _possible_values_list = []

                # We append k to attributes and it's possible values to _possible_values_list.
                attributes = [k]
                _possible_values_list.append(attribute_values[k])

                # Here we append it's parents to attributes and
                # all the possible values of k's parents to _possible_values_list.
                for value in v :
                    attributes.append(value)
                    _possible_values_list.append(attribute_values[value])

                # Now we get the probability for each combination of parent and k's values.

                it = itertools.product(*_possible_values_list)
                sums = defaultdict(float)
                for combination in it :
                     count_ocurrences(attributes, combination, data, positions, sums)

                probabilities = {}

                # We store the index of k attribute for future use.
                index = attributes.index(k)

                # Now we remove k attribute and it's values, to count
                # for the division. P(Bi|Aj,Ck) = #BiAjCk/ (#AjCk)
                # We are counting #AjCk now
                _possible_values_list.remove(attribute_values[k])
                attributes.remove(k)
                it = itertools.product(*_possible_values_list)
                sums_down = defaultdict(float)
                for combination in it :
                    count_ocurrences(attributes, combination, data, positions, sums_down)

                for key in sums:
                    key_c = tuple(elem for position, elem in enumerate(key) if position != index)
                    count = sums_down[key_c]
                    # We divide using Laplace correction
                    probabilities[key] = (sums[key] + 1) / (count + len(attribute_values[k]))

                conditional_probabilities.append(probabilities)


def predict(tuple):
    total = 0
    attribute_real_values= attribute_values.copy()

    # If attribute is set, then we don't have to use total probability theorem
    # over that attribute.
    for k, v in tuple.items():
        if v is not None:
            attribute_real_values[k] = [discretize(v,k)]


    for gre_value in attribute_real_values["GRE"] :
        for gpa_value in attribute_real_values["GPA"] :
            for admit_value in attribute_real_values["admit"]:
                for rank_value in attribute_real_values["rank"]:
                    aux = conditional_probabilities[0][(admit_value, gre_value, gpa_value, rank_value)]

                    # We check if there is more than 1 possible value
                    # If so, we have to multiply by its probability.
                    # If there is only one, then the probability is 1.
                    if len(attribute_real_values["GPA"]) > 1:
                        aux *=conditional_probabilities[2][(gpa_value, rank_value)]
                    if len(attribute_real_values["GRE"]) > 1:
                        aux *= conditional_probabilities[1][(gre_value, rank_value)]
                    if len(attribute_real_values["rank"]) > 1:
                        aux *= prior_probabilities[0][rank_value]

                    total += aux

    return total



def main():

    train()


    proba = predict({"admit": False,
                    "rank": 1})


    print("P(admit = false / rank = 1) = ", proba)

    proba = predict({"admit" : True,
                     "rank" : 2,
                     "GRE" : 450,
                     "GPA" : 3.5})

    print("P(admit = true / rank = 2, GRE = 450, GPA = 3.5) = ", proba)



if __name__ == '__main__':
    main()

