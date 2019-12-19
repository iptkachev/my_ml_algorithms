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
    def __init__(self, input_size, output_size, activate_fn_val):
        self.W = np.random.random((input_size, output_size))
        self.activate_fn_val = activate_fn_val
        if self.activate_fn_val == 'relu':
            self.activate_fn = ReLU.activate
            self.grad = ReLU.grad

        elif self.activate_fn_val == 'sigmoid':
            self.activate_fn = Sigmoid.activate
            self.grad = Sigmoid.grad

        elif self.activate_fn_val == 'linear':
            self.activate_fn = Linear.activate
            self.grad = Linear.grad

    def forward(self, X):
        print(X.shape, self.W.shape)
        return self.activate_fn(X @ self.W)

    def backward(self, X):
        return self.grad(X @ self.W)


class Net(NNObject):
    def __init__(self, loss_fn, *layers, **kwargs):
        self.layers = list(layers)
        self.random_state = kwargs.get('random_state', 10)
        self.loss_fn = loss_fn

    def _batch_generator(self, X, y, batch_size):
        np.random.seed(self.random_state)
        perm = np.random.permutation(len(X))
        n_batches = ceil(len(X) / batch_size)
        for batch_idx in range(n_batches):
            idx = perm[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            yield X[idx, :], y[idx]

    def forward(self, X, return_outputs=None):
        if return_outputs:
            outputs = []
        input = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            if return_outputs:
                outputs.append(input)
            input = layer.forward(input)
        if return_outputs:
            return input, outputs
        return input

    def backward(self, X, y, learning_rate):
        y_pred, layers_outputs = self.forward(X, True)
        loss = self.loss_fn.activate(y, y_pred)
        grad_val = self.loss_fn.grad(y, y_pred)
        print(loss)
        print(grad_val.shape)
        for i, layer in enumerate(reversed(self.layers)):
            if not i + 1 == len(self.layers):
                input = layers_outputs[-(i + 1)]
            else:
                input = X
            print('grad_val', grad_val.shape)
            print('input', input.shape)
            print('layer.W', layer.W.shape)
            print('layer.backward', layer.backward(input).shape)
            grad_val = input.T @ (layer.backward(input) * grad_val)
            print(grad_val.shape)
            layer.W -= learning_rate * grad_val

    def train(self, X, y, learning_rate, epochs, batch_size):
        for _ in tqdm(range(epochs)):
            for _ in tqdm(self._batch_generator(X, y, batch_size)):
                self.backward(X, y, learning_rate)