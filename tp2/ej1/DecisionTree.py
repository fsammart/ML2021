import json
import math
import os
from collections import defaultdict


class DecisionTree:
    _attributes = None
    _decision_tree = None
    result_variable = None
    name = None
    _features = None  # dont include target variable
    gain_function = None
    height_limit = None  # Height limit of the tree
    leaf_nodes = 0

    def __init__(self, data, attributes, result_variable, name, gain_function, height_limit=None):
        self.name = name
        self._attributes = attributes
        self._features = list(attributes)
        self._features.remove(result_variable)
        self.result_variable = result_variable
        self.gain_function = gain_function
        self.height_limit = height_limit

        self._decision_tree = self._make_tree(data, attributes, 0)
        self._dict_to_dot()

    def predict(self, observation):
        return self._predict(observation, self._decision_tree)

    def _predict(self, observation, subtree):

        # If Subtree is not a dict then its a leaf.
        if not isinstance(subtree, dict):
            return subtree
        attribute = list(subtree.keys())[0]

        if observation[attribute] in subtree[attribute]:
            return self._predict(observation, subtree[attribute][observation[attribute]])

        results = []
        # If it's not, we'll continue in every branch
        for attribute, values in subtree.items() :
            # For each value and subtree in the tree
            for value, value_subtree in values.items() :
                results.append(self._predict(observation, value_subtree))

        return max(results, key = results.count)



        #return self._default_feature_value[self.result_variable]

    # Default value is found by most occurrences
    def current_moda(self, attribute, data):

        val_freq_dict = defaultdict(int)

        for observation in data:
            val_freq_dict[observation[attribute]] += 1

        # Get the key with the max value
        # If there are several max values, return one (random)
        maximum = max(val_freq_dict, key=(lambda k: val_freq_dict[k]))
        return maximum


    def get_values(self, data, attr):
        return set([observation[attr] for observation in data])

    def _gain(self, attributes, data, attr):
        val_freq = defaultdict(float)
        subset_entropy = 0.0

        for observation in data:
            val_freq[observation[attr]] += 1.0

        # len of data, just another way
        total_values = sum(val_freq.values())

        for val in val_freq:
            val_prob = val_freq[val] / total_values
            # Get subset to calculate gain function
            data_subset = [entry for entry in data if entry[attr] == val]

            subset_entropy += val_prob * self.gain_function(data_subset, self.result_variable)

        gain = self.gain_function(data, self.result_variable) - subset_entropy
        return gain


    @staticmethod
    def entropy(data, target_variable):
        val_freq = defaultdict(float)
        _entropy = 0.0

        for entry in data:
            val_freq[entry[target_variable]] += 1.0

        for freq in val_freq.values():
            if freq != 0:
                _entropy += (-freq / len(data)) * math.log(freq / len(data), 2)

        return _entropy

    @staticmethod
    def gini(data, target_variable):
        val_freq = defaultdict(float)
        _sum = 0.0

        for entry in data:
            val_freq[entry[target_variable]] += 1.0

        for freq in val_freq.values():
            _sum += (freq / len(data)) ** 2

        return 1 - _sum

    def get_subtree_data(self, data, best, val):
        subtree_data = []
        for entry in data:
            if entry[best] == val:
                new_entry = defaultdict(str)
                for attribute in entry:
                    if attribute != best:
                        new_entry[attribute] = entry[attribute]
                subtree_data.append(new_entry)
        return subtree_data

    # Choose best attribute
    def find_best_feature(self, data, attributes):
        best = None
        max_gain = 0
        for attr in attributes:
            # Do not use result variable
            if attr == self.result_variable:
                continue

            current_gain = self._gain(attributes, data, attr)
            if current_gain >= max_gain:
                max_gain = current_gain
                best = attr
        return best

    def _make_tree(self, data, attributes, height):
        height += 1

        # Contains all target values of current data
        target_values = [observation[self.result_variable] for observation in data]


        # CASE A: If the height is maximum, or there are no attributes.
        if  (len(attributes) - 1) <= 0  or \
                (self.height_limit is not None and height > self.height_limit):
            self.leaf_nodes += 1
            return self.current_moda(self.result_variable, data)

        # CASE B: All observations have the same target value. Return that value.
        elif len(set(target_values)) == 1:
            self.leaf_nodes += 1
            return target_values[0]

        # CASE C: Here we expand the node.
        else:
            best = self.find_best_feature(data, attributes)
            # Create a new node based on best attribute
            tree = {best: {}}

            # Create branches for each value
            values_for_best_attribute = self.get_values(data, best)

            for val in values_for_best_attribute:
                # prepare data for subtree
                data_for_subtree = self.get_subtree_data(data, best, val)
                # prepare attributes
                new_attributes = list(attributes)
                new_attributes.remove(best)

                # Recursive
                subtree = self._make_tree(data_for_subtree, new_attributes, height)

                # Add the new subtree to the empty dictionary object in our new
                # tree/node we just created.
                tree[best][val] = subtree

        return tree

    ## Visual Representation

    def __str__(self):
        return f'{self.name} \n {json.dumps(self._decision_tree, indent=4)}'

    dot_representation = None

    def create_dot_image(self):
        self.save_to_dot()
        os.system(f"dot -Tpng {self.name}.dot -o {self.name}.png")

    def save_to_dot(self):
        with open(f"{self.name}.dot", "+w") as f:
            f.write(self.dot_representation)

    def _dict_to_dot(self):
        self.dot_representation = ''
        self.dot_representation += "strict digraph tree {\n"
        self._get_dot_edges(self._decision_tree, None)
        self.dot_representation += "}\n"

    def _get_dot_edges(self, subtree, parent):
        if not isinstance(subtree, dict):
            self.dot_representation += f'    "{subtree}" [label="{subtree}"];\n'
            return
        name = next(iter(subtree.keys()))
        label = name
        if parent:
            label = f'{parent}-{name}'
            self.dot_representation += f'    "{label}" [label="{name}"];\n'
            self.dot_representation += f'    "{parent}" -> "{label}";\n'
        for attr, value in subtree[name].items():
            if isinstance(value, dict):
                self._get_dot_edges({attr: value}, label)
            else:
                self.dot_representation += f'    "{label}-{attr}" [label="{attr}"];\n'
                self.dot_representation += f'    "{label}-{attr}-{value}" [label="{value}"];\n'
                self.dot_representation += f'    "{label}" -> "{label}-{attr}";\n'
                self.dot_representation += f'    "{label}-{attr}" -> "{label}-{attr}-{value}";\n'