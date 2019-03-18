import numpy as np
from perceptron import Perceptron

training_data = []
training_data.append(np.array([1, 1]))
training_data.append(np.array([1, 0]))
training_data.append(np.array([0, 1]))
training_data.append(np.array([0, 0]))

labels = np.array([0, 0, 0, 1])

perceptron = Perceptron(2, epochs=40)
_ = perceptron.train(training_data, labels)

print(perceptron.weights)

inputs = np.array([1, 1])
print(perceptron.predict(inputs)[0])

inputs = np.array([1, 0])
print(perceptron.predict(inputs)[0])

inputs = np.array([0, 1])
print(perceptron.predict(inputs)[0])

inputs = np.array([0, 0])
print(perceptron.predict(inputs)[0])
