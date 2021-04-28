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
    data_umbral_explode = None
    gain_umbral = None

    def __init__(self, data, attributes, result_variable,
                 name, gain_function,
                 height_limit=None,
                 data_umbral_explode=None,
                 gain_umbral=None):
        self.name = name
        self._attributes = attributes
        self._features = list(attributes)
        self._features.remove(result_variable)
        self.result_variable = result_variable
        self.gain_function = gain_function
        self.expanded_nodes = 0
        self.height_limit = height_limit
        self.data_umbral_explode = data_umbral_explode
        self.gain_umbral = gain_umbral
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

    # Select best attribute based on gain
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
        return best, max_gain

    def _make_tree(self, data, attributes, height):
        height += 1
        target_values = [observation[self.result_variable] for observation in data]

        explode = True

        # Check if there is enough data to divide. In case Poda is activated
        if self.data_umbral_explode is not None:
            if len(target_values) < self.data_umbral_explode:
                explode = False

        # CASE 1: If the height is maximum, or there are no attributes.
        if not explode or (len(attributes) - 1) <= 0  or \
                (self.height_limit is not None and height > self.height_limit):
            self.leaf_nodes += 1
            return self.current_moda(self.result_variable, data)

        # CASE 2: All observations have the same target value. Return that value.
        elif len(set(target_values)) == 1:
            self.leaf_nodes += 1
            return target_values[0]

        # CASE 3: Here we expand the node.
        else:
            self.expanded_nodes += 1
            best, gain = self.find_best_feature(data, attributes)
            # Check if gain is enough to divide
            if self.gain_umbral is not None:
                if gain < self.gain_umbral:
                    self.leaf_nodes += 1
                    return self.current_moda(self.result_variable, data)

            # Create new node to expand
            tree = {best: {}}

            # Create branches for each value
            values_for_best_attribute = self.get_values(data, best)

            for val in values_for_best_attribute:
                # prepare data to create subtree
                data_for_subtree = self.get_subtree_data(data, best, val)
                new_attributes = list(attributes)
                new_attributes.remove(best)
                # Call the recursive function
                subtree = self._make_tree(data_for_subtree, new_attributes, height)
                # We add our result
                tree[best][val] = subtree

        return tree


    # entropy functions

    @staticmethod
    def gini(data, target_variable):
        freq_variable = defaultdict(float)
        sum = 0.0

        for entry in data:
            freq_variable[entry[target_variable]] += 1.0

        for freq in freq_variable.values():
            sum += (freq / len(data)) ** 2

        return 1 - sum

    @staticmethod
    def shannon(data, target_variable):
        freq_variable = defaultdict(float)
        entropy = 0.0

        for entry in data:
            freq_variable[entry[target_variable]] += 1.0

        for freq in freq_variable.values():
            if freq != 0:
                entropy += (-freq / len(data)) * math.log(freq / len(data), 2)

        return entropy
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