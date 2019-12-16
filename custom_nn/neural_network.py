from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from math import ceil
from custom_nn.utils import *

__all__ = ['Net', 'Layer']


class NNObject(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


class Layer(NNObject):
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


class InitLayer(NNObject):
    def forward(self, x):
        return x

    def backward(self, x):
        return x


class Net(NNObject):
    def __init__(self, loss_fn, *layers, **kwargs):
        self.layers = [InitLayer()] + list(layers)
        self.random_state = kwargs.get('random_state', 10)
        self.loss_fn = loss_fn

    def _batch_generator(self, X, y, batch_size):
        np.random.seed(self.random_state)
        perm = np.random.permutation(len(X))
        n_batches = ceil(len(X) / batch_size)
        for batch_idx in range(n_batches):
            idx = perm[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            yield X[idx, :], y[idx]

    def forward(self, X):
        input = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            input = layer.forward(input)

        return input

    def backward(self, X, y, learning_rate):
        y_pred = self.forward(X)
        loss = self.loss_fn.activate(y, y_pred)
        grad_val = self.loss_fn.grad(y, y_pred)
        print(loss)
        print(grad_val.shape)
        for layer in reversed(self.layers):
            grad_val *= layer.backward(X)
            print(layer.backward(X).shape)
            print(grad_val.shape)
            layer.W -= learning_rate * grad_val

    def train(self, X, y, learning_rate, epochs, batch_size):
        for _ in tqdm(range(epochs)):
            for _ in tqdm(self._batch_generator(X, y, batch_size)):
                self.backward(X, y, learning_rate)