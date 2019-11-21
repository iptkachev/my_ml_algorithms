import numpy as np


class PairLinearRegressian:
    def __init__(self, initial_w=[0, 0], learning_rate=1e-06, iters=10000, eps=1e-05, optimizer=None):
        self.coef = np.array(initial_w, dtype='float64')
        self.learning_rate = learning_rate
        self.iters = iters
        self.eps = eps
        self._fitted = None
        if optimizer is None:
            self.optimizer = 'GD'
        else:
            self.optimizer = optimizer

    def _check_dim(self, X):
        if X.shape[1] == 1:
            return np.hstack((np.array([1] * X.shape[0]).reshape(-1, 1), X))
        elif X.shape[1] == 2:
            return X
        else:
            raise ValueError

    def _grad_intercept(self, X, y, coef, random_index=None):
        if random_index is None:
            return np.mean(X @ coef - y)
        else:
            return X[random_index] @ coef - y[random_index]

    def _grad_slope(self, X, y, coef, random_index=None):
        if random_index is None:
            return np.mean(X.T * (X @ coef - y))
        else:
            return X[random_index, 1] * (X[random_index] @ coef - y[random_index])

    def fit(self, X, y):
        X = self._check_dim(X)
        for i in range(self.iters):
            previous_coef = np.copy(self.coef)

            if self.optimizer == 'GD':
                self.coef[0] -= self.learning_rate * self._grad_intercept(X, y, self.coef)
                self.coef[1] -= self.learning_rate * self._grad_slope(X, y, self.coef)
            elif self.optimizer == 'SGD':
                random_index = np.random.randint(0, X.shape[0], size=1)
                self.coef[0] -= self.learning_rate * self._grad_intercept(X, y, self.coef, random_index)
                self.coef[1] -= self.learning_rate * self._grad_slope(X, y, self.coef, random_index)

            if (np.abs(self.coef - previous_coef) < self.eps).all():
                break
        print(i)
        self._fitted = True

    def predict(self, X):
        X = self._check_dim(X)
        if self._fitted:
            return X @ self.coef
