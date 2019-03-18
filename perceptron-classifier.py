import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron


def get_data(file_path, delimiter=','):
    return np.genfromtxt(file_path, delimiter=delimiter)


def visualize_data(data):
    class_one_data = np.array([[d[0],d[1]] for d in data if d[2]==0])
    class_two_data = np.array([[d[0],d[1]] for d in data if d[2]==1])
    class_one_x = np.array([d[0] for d in class_one_data])
    class_one_y = np.array([d[1] for d in class_one_data])
    class_two_x = np.array([d[0] for d in class_two_data])
    class_two_y = np.array([d[1] for d in class_two_data])
    plt.scatter(class_one_x, class_one_y, marker='^', color='r')
    plt.scatter(class_two_x, class_two_y, marker='o', color='b')


data = get_data('data.txt')
visualize_data(data)
training_data = np.array([[d[0],d[1]] for d in data])
labels = np.array([d[2] for d in data])
perceptron = Perceptron(2, epochs=2000)
errors = perceptron.train(training_data, labels)


x = np.arange(-200, -50, 0.01)
y = -(perceptron.weights[1]/perceptron.weights[2]) * x - (perceptron.weights[0]/perceptron.weights[2])
plt.plot(x,y)
plt.show()
plt.plot(errors)
plt.show()




