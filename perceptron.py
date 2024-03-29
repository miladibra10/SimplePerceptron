import numpy as np


class Perceptron:
    """
    Implementation of perceptron with gradient decent algorithm for learning and updating weights
    """
    def __init__(self, input_number, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(input_number + 1)

    def predict(self, inputs):
        # weight[0] is bias
        sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if sum > 0:
            activation = 1
        else:
            activation = 0
        return activation, sum

    def train(self, training_inputs, labels):
        mse = []
        for _ in range(self.epochs):
            squerd_loss = []
            for inputs, label in zip(training_inputs, labels):
                prediction, sum = self.predict(inputs)
                loss = label - prediction
                squerd_loss.append(loss**2)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
            mse.append(np.mean(squerd_loss))
        return mse
