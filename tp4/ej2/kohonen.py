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

    def activations(self, filename):
        activations = np.zeros((self.k, self.k))

        for sample in self.data:
            winner = self.predict_winner(sample)
            activations[winner[0]][winner[1]] += 1

        plt.imshow(np.squeeze(activations), cmap="Blues")
        plt.colorbar()
        plt.title("Activations count for each neuron.")
        plt.savefig(f'./results/{filename}')
        plt.clf()

    def u_matrix(self, filename):
        u_matrix = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                neighbours = self.get_neighbours(i, j, math.sqrt(2))
                neigh_weights = np.fromiter(map(
                    lambda x: self.kohonen[x], neighbours
                ), dtype=np.float64)
                distances = np.fromiter(map(
                    euclidean(self.kohonen[(i, j)]), neigh_weights
                ), dtype=np.float64)
                u_matrix[i][j] = np.mean(distances)

        plt.imshow(np.squeeze(u_matrix), cmap="Greys")
        plt.colorbar()
        plt.title("U Matrix for SOM")
        plt.savefig(f'./results/{filename}')
        plt.clf()
