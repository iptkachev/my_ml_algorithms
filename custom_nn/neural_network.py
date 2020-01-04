from abc import ABC, abstractmethod
from tqdm import tqdm
from math import ceil
from pandas import DataFrame
import numpy as np

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
        self.W = np.random.uniform(-1, 1, (input_size, output_size))
        self.activate_fn = activate_fn.activate
        self.grad = activate_fn.grad

    def forward(self, X):
        return self.activate_fn(X @ self.W)

    def backward(self, X):
        return self.grad(X @ self.W)


class Net(NNObject):
    def __init__(self, loss_fn, random_state):
        self.layers = []
        self.random_state = random_state
        self.loss_fn = loss_fn

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _batch_generator(self, X, y, batch_size):
        np.random.seed(self.random_state)
        perm = np.random.permutation(len(X))
        n_batches = ceil(len(X) / batch_size)
        for batch_idx in range(n_batches):
            idx = perm[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            yield X[idx, :], y[idx]

    def _check_intercept(self, X):
        ones_vector = np.ones((X.shape[0], 1))
        if isinstance(X, DataFrame):
            X = X.values
        if (X[:, [0]] == ones_vector).all():
            return X
        else:
            return np.hstack((ones_vector, X))

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X, return_outputs=None):
        if return_outputs:
            outputs = []
            input = self.layers[0].forward(X)
            for layer in self.layers[1:]:
                outputs.append(input)
                input = layer.forward(input)
            return input, outputs
        else:
            input = self.layers[0].forward(X)
            for layer in self.layers[1:]:
                input = layer.forward(input)
            return input

    def backward(self, X, y, learning_rate):
        y_pred, layers_outputs = self.forward(X, return_outputs=True)
        layers_outputs = layers_outputs[::-1]
        layers_outputs.append(X)
        reversed_layers = self.layers[::-1]
        grad_val = self.loss_fn.grad(y, y_pred) * reversed_layers[0].backward(layers_outputs[0])
        for i, layer in enumerate(reversed_layers):
            input = layers_outputs[i]
            if i > 0:
                grad_val = grad_val @ reversed_layers[i - 1].W.T * layer.backward(input)
            grad_W = input.T @ grad_val
            layer.W -= learning_rate * grad_W

    def train(self, X, y, learning_rate: float, epochs: int, batch_size: int, verbose=10):
        for e in tqdm(range(epochs)):
            for b, (X_batch, y_batch) in enumerate(tqdm(self._batch_generator(X, y, batch_size))):
                self.backward(X_batch, y_batch, learning_rate)
                if e * b % verbose == 0 and e * b >= verbose:
                    print('Training loss', self.loss_fn.activate(y, self.forward(X)))