import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator


class MyLogisticRegression(BaseEstimator):
    def __init__(self, learning_rate=0.05, penalty='l2', C=10000.0, iters=10000, tol=1e-4,
                 multi_class='binary', optimizer=None):
        self._coef = None
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.iters = iters
        self.tol = tol
        self.multi_class = multi_class
        self._fitted = None

        if optimizer is None:
            self.optimizer = 'SGD'
        else:
            self.optimizer = optimizer
        if penalty == 'l1':
            self.alpha = 0.
            self.beta = 1. / C
        else:
            self.alpha = 1. / C
            self.beta = 0.

    def _check_intercept(self, X):
        ones_vector = np.ones((X.shape[0], 1))
        if isinstance(X, pd.DataFrame):
            X = X.values
        if (X[:, [0]] == ones_vector).all():
            return X
        else:
            return np.hstack((ones_vector, X))

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @property
    def coef(self):
        return self._coef

    def _grad_loss(self, X, y, random_index=None):
        if random_index is None:
            return X.T @ (self._sigmoid(X @ self._coef.T) - y) / X.shape[0]
        else:
            return X[[random_index]].T @ (self._sigmoid(X[[random_index]] @ self._coef.T) - y[[random_index]])

    def fit(self, X, y):
        X = self._check_intercept(X)
        if self.multi_class == 'binary':
            if isinstance(y, pd.Series):
                y = y.values.reshape(-1, 1)
            self._coef = np.zeros((1, X.shape[1]))
        elif self.multi_class == 'multilabel':
            y = pd.get_dummies(y).values  # n x K
            self._coef = np.zeros((y.shape[1], X.shape[1]))  # K x m

        for i in range(self.iters):
            previous_coef = np.copy(self._coef)
            random_index = np.random.randint(0, X.shape[0], size=1)
            self._coef -= self.learning_rate * \
                          self._grad_loss(X, y,
                                          random_index if self.optimizer == 'SGD' else None).T + \
                          self.alpha * self._coef + self.beta * np.abs(self._coef)
            if (np.abs(self._coef - previous_coef) < self.tol).all():
                break
        print(i)
        self._fitted = True

        return self

    def predict_proba(self, X):
        X = self._check_intercept(X)
        if self._fitted:
            if self.multi_class == 'binary':
                return self._sigmoid(X @ self.coef.T).reshape(-1)
            elif self.multi_class == 'multilabel':
                return self._sigmoid(X @ self.coef.T)
        else:
            raise NotFittedError

    def predict(self, X):
        prob = self.predict_proba(X)
        if self.multi_class == 'binary':
            return prob.round()
        elif self.multi_class == 'multilabel':
            maxes = prob.max(1).reshape(-1, 1)
            return np.where(prob == maxes)[1]
