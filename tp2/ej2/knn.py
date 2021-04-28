import math

import numpy as np


def euclidean_distance_to(x1):
    def euclidean_distance(x2):
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i]) ** 2
        return math.sqrt(distance)
    return euclidean_distance


class KNN:

    def __init__(self, data, labels, classes, k=5):
        self.k = k
        self.data = data
        self.labels = labels
        self.classes = classes

    @staticmethod
    def weight(distance):
        return 1  # you are just counting

    def predict(self, register):
        # computes distance from register to existing data
        distances = list(map(euclidean_distance_to(register), self.data))

        # joins distances with corresponding classes
        distances_with_classes = list(zip(distances, self.labels))

        # selects first k neighbours
        distances_with_classes.sort()
        knn = distances_with_classes[:self.k]

        # if register is the same as any of training data, will return class
        if knn[0][0] == 0.0: return knn[0][1]

        # computes max class
        results = np.zeros(len(self.classes))
        for (x,y) in knn: results[y-1] += self.weight(x) # x: distance, y: class

        max_value = np.max(results)
        winner = np.where(results == max_value)[0]
        return winner[0]+1  if len(winner) == 1 or self.k == len(self.data) else self.untie(register)

    def batch_predict(self, batch):
        return list(map(self.predict, batch))

    def untie(self, register):
        k = self.k + 1
        knn = self.__class__(self.data, self.labels, self.classes, k=k)
        return knn.predict(register)


class WeightedKNN(KNN):

    @staticmethod
    def weight(distance):
        return 1/(distance ** 2) if distance != 0 else math.inf


class CustomWeightedKNN(KNN):

    @staticmethod
    def weight(distance):
        return 1/abs(distance) if distance != 0 else math.inf

