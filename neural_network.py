from abc import ABC, abstractmethod
import numpy as np
from activate_functions import *

__all__ = ['Net', 'Layer']


class Neuron(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


class Layer(Neuron):
    def __init__(self, input_size, output_size, activate_fn):
        self.W = np.random.random((input_size, output_size))
        if activate_fn == 'relu':
            self.activate_fn = ReLU.activate
            self.grad = ReLU.grad

        elif activate_fn == 'sigmoid':
            self.activate_fn = Sigmoid.activate
            self.grad = Sigmoid.grad

        elif activate_fn == 'linear':
            self.activate_fn = Linear.activate
            self.grad = Linear.grad

    def forward(self, X):
        return self.activate_fn(X @ self.W)

    def backward(self, X):
        return self.grad(X @ self.W)


class InitLayer(Neuron):
    def forward(self, x):
        return x

    def backward(self, x):
        return x


class Net(Neuron):
    def __init__(self, learning_rate, *layers):
        self.layers = [InitLayer] + list(layers)
        self.learning_rate = learning_rate

    def forward(self, X):
        input = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            input = layer(input)

        return X

    def backward(self, X, y):
        grad_val = self.layers[0].grad(X)
        self.layers[0].W -= self.learning_rate * grad_val
        for layer in self.layers[1:]:
            grad_val *= layer(input)
            layer.W -= self.learning_rate * grad_val

        return X
