import numpy as np
import matplotlib.pyplot as plt
import random
from tp3.ej1.utils import shortest_distance, calculate_margin, calculate_correctness
import itertools

from numpy.core.shape_base import block



class SimplePerceptron:
    def __init__(self, data, max_epoch=1000, learning_rate=1, calculate_errors=False, visualize=False):
        self.max_epoch = max_epoch
        self.samples = data.shape[0]
        self.raw_data = data
        self.raw_data = np.hstack((self.raw_data, np.zeros((self.samples,1))))
        #We are assuming last column of data is the class
        self.attr_size = data.shape[1] - 1
        self.data = data[:,0:self.attr_size] #Remove the class attribute
        self.data = np.hstack((self.data,np.ones((self.samples, 1)))) # We add the independent term
        self.classes =  data[:, self.attr_size]
        self.weights = np.ones((self.attr_size + 1))
        self.learning_rate = learning_rate
        self.last = None
        self.calculate_errors = calculate_errors
        if self.calculate_errors:
            self.visualize = visualize
        else:
            if visualize:
                print("Cannot run visualization with error calculation off")
                self.visualize = False


    def plot_classes(self):
        c1 = self.raw_data[self.raw_data[:,2] == 1]
        c2 = self.raw_data[self.raw_data[:,2] == -1]
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
        plt.ion()
        plt.show(block=False)
        

    # In case we want to change activation function
    def activate(self, input):
        return np.sign(input)

    def train(self):
        self.plot_classes()
        epoch = 0
        error = 1
        min_error = 4*self.samples
        while error > 0 and epoch < self.max_epoch:
            r_index = random.randint(0,self.samples - 1)
            sample = self.data[r_index,:]
            sample_class = self.classes[r_index]
            excitation = self.weights.dot(sample)
            activation = self.activate(excitation)
            epoch += 1

            local_error = sample_class - activation
            delta_w = local_error * self.learning_rate * sample
            self.weights = self.weights + delta_w
    
            
            # Update weights and error if error < min_error
            # This hugely slows runtime, as it increases complexity from O(T) to O(T*N)
            # where T is epochs and N is sample size

            if self.calculate_errors:
                error = self.calculate_error()
                if error < min_error:
                    min_error = error
                    min_weights = self.weights
                
                if self.visualize:
                    self.draw()
        if self.calculate_errors:
            self.weights = min_weights
            self.error = min_error
        else:
            self.error = self.calculate_error()

    def draw_with_slope(self, m, ind):
        x = np.array((-1, 6))
        y = np.array((-1 * m + ind, 6 * m + ind))

        if self.last :
            l = self.last.pop(0)
            l.remove()
        self.last = plt.plot(x, y, color="black")
        colors = np.where(self.raw_data[:, 3] > 0, 'red', 'blue')
        plt.scatter(self.raw_data[:, 0], self.raw_data[:, 1], color=colors)
        plt.show(block=False)
        plt.pause(0.01)

    def draw(self):
        a,b,c = self.weights[0], self.weights[1], self.weights[2]
        if b == 0:
            return
        m = -1 * a / b
        ind = -1 * c / b
        self.draw_with_slope(m,ind)

    def calculate_error(self):
        class_col = self.attr_size
        pred_col = self.attr_size + 1
        for i in range(self.samples):
            self.raw_data[i,pred_col] = self.predict(self.raw_data[i,0:class_col])
        
        error = self.raw_data[:,class_col] - self.raw_data[:,pred_col]

        error = error **2
        error_sum = np.sum(error)
        return error_sum



    def predict(self, sample):
        sample = np.hstack((sample,1))
        excitation = self.weights.dot(sample)
        activation = self.activate(excitation)
        return activation

    ## Call after training.
    ## n: number of points on each side of the plane to check. (4)
    ## k: number of support vectors.
    def optimus_hiperplane(self, n=4):
        k=3
        # Calculate current hiperplane
        a, b, c = self.weights[0], self.weights[1], self.weights[2]

        # Get n closest elements from each class to the current hiperplane
        data2 = list(self.raw_data.copy())
        data2.sort(key=lambda t: shortest_distance(t[0],t[1],a,b,c))
        # data2 is now sorted, get n elements from each class
        class1 = []
        class2 = []
        idx = 0
        while idx < len(data2) and (len(class1)<n or len(class2) <n):
            curr_class = data2[idx][3]
            if curr_class == 1 and len(class1)<n:
                class1.append(data2[idx])
            if curr_class == -1 and len(class2)<n:
                class2.append(data2[idx])
            idx +=1

        # Then get k elements in total from 2*n and check best combination.
        # floor(k/2) from class1 and ceil(k/2) from class2
        k1 = 2
        k2 = 1
        final_margin = 0
        final_m = -1 * a / b
        final_b = -1 * c / b
        final_points = None
        final_distance = None
        for ck1 in itertools.combinations(class1, k1):
            for ck2 in itertools.combinations(class2, k2):
                # here we have the combinations.
                # Dummy code so that it doesn´t mark as error
                p1 = ck1[0][0:2]
                p2 = ck1[1][0:2]
                p3 = ck2[0][0:2]
                line = p2 - p1
                line = line/np.linalg.norm(line)
                slope = line[1]/line[0]
                d1 = np.linalg.norm(p3-p1)
                d2 = np.linalg.norm(p3-p2)
                chosen = None
                if d1<d2:
                    chosen = p1
                else:
                    chosen = p2
                mpoint = [(chosen[0] + p3[0])/2,(chosen[1] + p3[1])/2]

                b = mpoint[1] - slope * mpoint[0]
                if not calculate_correctness(slope,b, data2):
                    continue
                curr_margin = calculate_margin(slope, b, data2)
                if curr_margin > final_margin:
                    final_m = slope
                    final_b = b
                    final_points = [p1,p2,p3]
                    final_margin=curr_margin

        for ck1 in itertools.combinations(class2, k1) :
            for ck2 in itertools.combinations(class1, k2) :
                # here we have the combinations.
                # Dummy code so that it doesn´t mark as error
                p1 = ck1[0][0 :2]
                p2 = ck1[1][0 :2]
                p3 = ck2[0][0 :2]
                line = p2 - p1
                line = line / np.linalg.norm(line)
                slope = line[1] / line[0]
                d1 = np.linalg.norm(p3 - p1)
                d2 = np.linalg.norm(p3 - p2)
                chosen = None
                if d1 < d2 :
                    chosen = p1
                else :
                    chosen = p2
                mpoint = [(chosen[0] + p3[0]) / 2, (chosen[1] + p3[1]) / 2]

                b = mpoint[1] - slope * mpoint[0]
                if not calculate_correctness(slope, b, data2) :
                    continue
                curr_margin = calculate_margin(slope, b, data2)
                if curr_margin > final_margin :
                    final_m = slope
                    final_b = b
                    final_points = [p1, p2, p3]
                    final_margin = curr_margin

        return final_m, final_b, final_margin, final_points





# ax + by + c = 0
# y = -a/bx - c/b