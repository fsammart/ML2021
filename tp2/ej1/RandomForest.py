import random
from tp2.ej1.DecisionTree import DecisionTree
import numpy as np

class RandomForest:

    trees = None

    def __init__ (self, data, attributes, target_variable, name, gain_function, number_of_trees,
                  number_of_attributes, number_of_elements,
                  height_limit=None) :
        self.trees = []
        features = list(attributes)
        features.remove(target_variable)
        for i in range(0, number_of_trees) :
            attributes_for_tree = random.sample(features, number_of_attributes)
            attributes_for_tree.append(target_variable)
            #Use choices for replacement
            train_data = random.choices(data, k=number_of_elements)
            tree = DecisionTree(train_data, attributes_for_tree, target_variable, name + "_" + str(i), gain_function,
                                height_limit)
            self.trees.append(tree)
        self.expanded_nodes = np.mean([tree.expanded_nodes for tree in self.trees])


    # We predict Aggregating Result, in this case moda
    def predict (self, observation) :
        results = []
        for tree in self.trees:
            results.append(tree.predict(observation))
        return max(results, key=results.count)


    ## Visualization Utils

    def create_dot_image (self) :
        for tree in self.trees:
            tree.create_dot_image()