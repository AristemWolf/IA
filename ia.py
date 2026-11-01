import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt

lr = 1e-3

def sigmoid(x : float) -> float:
    return 1 / (1 + math.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation = True):
        self.w = np.random.randn(n_neurons, n_inputs) * np.sqrt(2 / n_inputs)
        self.b = np.zeros((n_neurons, 1))
        self.activation = activation

    def forward(self, x):
        self.input = x
        z = self.w @ x + self.b
        self.z = z
        if self.activation:
            a = relu(z)
            self.output = a
            return a
        self.output = z
        return z

    def backward(self, grad_output):
        if self.activation:
            dz = grad_output * relu_derivative(self.z)
        else:
            dz = grad_output

        self.dw = dz @ self.input.T
        self.db = dz
        dx = self.w.T @ dz

        return dx

    def update(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

class NeuralNetwork:
    def __init__(self, sizes):
        self.layers = []
        for i in range(len(sizes)-1):
            activation = (i != len(sizes)-2)
            layer = DenseLayer(
                sizes[i],
                sizes[i+1],
                activation = activation
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

nn = NeuralNetwork([1, 10, 100, 10, 1])

X = np.linspace(-1, 1, 1000)
Y = X ** 2

for epoch in range(1000):
    plt.clf()
    predictions = []
    total_loss = 0
    for i in range(1000):
        x = np.array([[X[i]]])
        y = np.array([[Y[i]]])
        pred = nn.forward(x)
        loss = mse(pred, y)
        total_loss += loss
        grad = 2 * (pred - y)
        nn.backward(grad)
        nn.update(lr)
        predictions.append(pred.item())
    # Vraie courbe
    plt.plot(X, Y, label="True")
    # prédictions réseau
    plt.plot(X, predictions, label="Pred")
    # plt.legend()
    plt.pause(0.01)
    print(epoch, total_loss / 1000)
