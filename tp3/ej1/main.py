import numpy as np
import matplotlib.pyplot as plt
from simple_perceptron import SimplePerceptron
from math import sqrt


def generate_separable_points(samples=10, margin=0):
    points = np.random.rand(samples, 2) * 5

    c1= points[points[:,1] > points[:,0]]
    c2= points[points[:,1] < points[:,0]]
    return c1,c2

def plot_classes(c1,c2):
    fig, ax = plt.subplots()
    plt.scatter(c1[:,0], c1[:,1],color="red")
    plt.scatter(c2[:,0], c2[:,1],color="blue")
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.show()



def get_train_data(c1,c2):
    c1 = np.hstack((c1,np.ones((c1.shape[0],1)) * -1))
    c2 = np.hstack((c2,np.ones((c2.shape[0],1)) * 1))
    data = np.vstack((c1,c2))
    return data

def main():
    print("Ejercicio 1 - Perceptron Simple")
    c1,c2 = generate_separable_points(100)
    data = get_train_data(c1,c2)
    
    sp = SimplePerceptron(data,max_epoch=1000,learning_rate=0.1, visualize=True, calculate_errors=True)
    sp.train()

    data = np.hstack((data,np.zeros((data.shape[0],1))))
    for i in range(data.shape[0]):
        data[i,3] = sp.predict(data[i,0:2])

    print("Error: ", sqrt(sp.error))
    # plot_classes(c1,c2)

if __name__ == "__main__":
    main()