from sklearn.datasets import make_blobs
from kohonen import Kohonen


feats = 1
k = 2
eta = 0.1
epochs = 100
data, y = make_blobs(n_samples=1000, n_features=feats, centers=3)
som = Kohonen(k, k, eta, feats, data, rand_weights=False)
som.train(epochs)
som.u_matrix('u_matrix.png')
som.activations('activations.png')