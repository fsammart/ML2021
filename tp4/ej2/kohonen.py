import random
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import get_indices_of_k_smallest, euclidean


def init_neuron(rand_weights, n_features, data):
    def f(x):
        if rand_weights:
            return np.random.rand(n_features)
        return data[random.randint(0, len(data)-1)]
    return f

class Kohonen:

    def __init__(self, k, R, eta, n_features, data, rand_weights=True):
        self.k = k
        self.init_R = R
        self.init_eta = eta
        self.n_features = n_features
        self.data = data
        self.matrix = (k, k, n_features)
        self.plain = (self.k*self.k, self.n_features)
        self.neurons = list(map(init_neuron(rand_weights, n_features, data), np.zeros(k*k)))
        self.kohonen = np.reshape(self.neurons, self.matrix)

        self.eta = None
        self.R = None
        self.cluster_classifications = None
        self.cluster_counts = None

    def set_radio(self, epoch, epochs):
        self.R = (epochs-epoch)*self.init_R/epochs

    def set_lr(self, epoch, epochs):
        self.eta = self.init_eta * (1-epoch/epochs)

    # TODO: improve method
    def get_neighbours(self, x, y, radio):
        neighbours = []
        distance = euclidean(np.array([x,y]))
        for i in range(self.k):
            for j in range(self.k):
                if distance(np.array([i, j])) <= radio:
                    neighbours.append((i, j))
        return neighbours

    def update_neuron(self, neuron, sample, from_winner_f):
        distance = sample - self.kohonen[neuron]
        V = math.exp(-2 * from_winner_f(np.array(neuron)) * (1/self.R))
        self.kohonen[neuron] = self.kohonen[neuron] + distance * self.eta * V

    def predict_winner(self, sample):
        distances = np.fromiter(map(euclidean(sample), self.kohonen.reshape(self.plain)), dtype=np.float64) \
            .reshape(self.k, self.k)
        return get_indices_of_k_smallest(distances, 1)

    def run_sample(self):
        sample = self.data[random.randint(0, len(self.data)-1)]
        winner = self.predict_winner(sample)
        neighbours = self.get_neighbours(winner[0], winner[1], self.R)
        from_winner_f = euclidean(np.array(winner))
        for n in neighbours:
            self.update_neuron(n, sample, from_winner_f)

    def train(self, epochs):
        for epoch in range(epochs):
            self.set_radio(epoch, epochs)
            self.set_lr(epoch, epochs)
            self.run_sample()

    def get_activations(self):
        activations = np.zeros((self.k, self.k))

        for sample in self.data:
            winner = self.predict_winner(sample)
            activations[winner[0]][winner[1]] += 1

        return activations

    def add_cluster_classification(self, labels):
        self.cluster_counts = np.zeros(shape=(self.k, self.k, 2))

        for i, sample in enumerate(self.data):
            winner = self.predict_winner(sample)
            self.cluster_counts[winner[0]][winner[1]][labels[i]] += 1

        self.cluster_classifications = np.zeros(shape=(self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                if self.cluster_counts[i][j][0] >= self.cluster_counts[i][j][1]:
                    self.cluster_classifications[i][j] = 0
                else:
                    self.cluster_classifications[i][j] = 1

    def predictor(self, sample):
        winner = self.predict_winner(sample)
        return int(self.cluster_classifications[winner[0]][winner[1]])

    def predict(self, samples):
        return list(map(self.predictor, samples))

    def plot_classification_som(self, filename):
        fig, ax = plt.subplots()
        ax.matshow(self.cluster_classifications, cmap=plt.cm.Set3, vmin=0, vmax=1)

        for i, j in np.ndindex(self.cluster_classifications.shape):
            decision = self.cluster_classifications[i][j]
            count = self.cluster_counts[i][j][0] + self.cluster_counts[i][j][1]
            probabilities = f'0 ->{round(self.cluster_counts[i][j][0]/count, 2)}\n1 ->{round(self.cluster_counts[i][j][1]/count, 2)}'
            ax.text(j, i, f'{decision}\n{probabilities}', va='center', ha='center')

        plt.title("Dominant class for each neuron")
        plt.savefig(f'./results/{filename}', bbox_inches='tight')
        plt.clf()

    def plot_activations(self, filename):
        activations = self.get_activations()

        fig, ax = plt.subplots()
        ax.matshow(activations, cmap=plt.cm.Blues, vmin=0)

        for i, j in np.ndindex(self.cluster_classifications.shape):
            value = f'{activations[i][j]}'
            ax.text(j, i, f'{value}', va='center', ha='center')

        plt.title("Activations count for each neuron.")
        plt.savefig(f'./results/{filename}', bbox_inches='tight')
        plt.clf()

    # def u_matrix(self, filename):
    #     u_matrix = np.zeros((self.k, self.k))
    #     for i in range(self.k):
    #         for j in range(self.k):
    #             neighbours = self.get_neighbours(i, j, math.sqrt(2))
    #             neigh_weights = np.fromiter(map(
    #                 lambda x: self.kohonen[x], neighbours
    #             ), dtype=np.float64)
    #             distances = np.fromiter(map(
    #                 euclidean(self.kohonen[(i, j)]), neigh_weights
    #             ), dtype=np.float64)
    #             u_matrix[i][j] = np.mean(distances)
    #
    #     plt.imshow(np.squeeze(u_matrix), cmap="Greys")
    #     plt.colorbar()
    #     plt.title("U Matrix for SOM")
    #     plt.savefig(f'./results/{filename}')
    #     plt.clf()
