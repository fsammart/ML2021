import numpy as np
import matplotlib.pyplot as plt
from simple_perceptron import SimplePerceptron, Slope
from utils import shortest_distance
from math import sqrt
from sklearn import svm


def generate_separable_points(samples=10, threshold=0, misclassifications=0):
    points = np.random.rand(samples, 2) * 5

    c1= points[points[:,1] > points[:,0] + threshold]
    c2= points[points[:,1] < points[:,0] - threshold]

    if misclassifications > 0:
        #sort by distance to x=y line

        c1 = list(c1)
        c2 = list(c2)

        c1.sort(key=lambda t: shortest_distance(t[0],t[1],1,-1,0))
        c2.sort(key=lambda t: shortest_distance(t[0],t[1],1,-1,0))
        c1 = np.array(c1)
        c2 = np.array(c2)
        tmp = np.vstack((c1[misclassifications:,:], c2[0:misclassifications, :]))
        c2 = np.vstack((c2[misclassifications:,:], c1[0:misclassifications, :]))
        c1 = tmp

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
    # plt.pause(0.00001)



def get_train_data(c1,c2):
    c1 = np.hstack((c1,np.ones((c1.shape[0],1)) * -1))
    c2 = np.hstack((c2,np.ones((c2.shape[0],1)) * 1))
    data = np.vstack((c1,c2))
    return data

def main():
    print("Ejercicio 1 - Perceptron Simple")
    c1,c2 = generate_separable_points(50,0.5, 2)
    # plot_classes(c1,c2)
    data = get_train_data(c1,c2)
    
    sp = SimplePerceptron(data,max_epoch=10000,learning_rate=0.01, visualize=False, calculate_errors=True)
    sp.train()

    data = np.hstack((data,np.zeros((data.shape[0],1))))
    for i in range(data.shape[0]):
        data[i,3] = sp.predict(data[i,0:2])

    print("Error: ", sqrt(sp.error))
    print("Weights: ", sp.weights)
    
    # sp.draw("Perceptron simple", "yellow")
    # plot_classes(c1,c2)
    m,b, margin, points = sp.optimus_hiperplane(n=4)

    perceptron_slope = sp.get_slope("Perceptron", "yellow")
    optimum_slope = Slope(m,b,"Hiperplano Optimo","green")
    if points is not None:
        sp.draw_with_slope([perceptron_slope, optimum_slope], False, np.array(points) + 0.01)
        print('final points', points)
    else: 
        sp.draw_with_slope([perceptron_slope, optimum_slope], False)


    classifier = svm.SVC(C=1, kernel='linear')
    clf = classifier.fit(data[:,0:2], data[:,2])
    pred = classifier.predict(data[:,0:2])
    plt.scatter(data[:, 0], data[:, 1], c=data[:,2], s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


if __name__ == "__main__":
    main()