import numpy as np
from functools import partial
from scipy.optimize import minimize
from sklearn.base import BaseEstimator


class CustomSVC(BaseEstimator):
    def __init__(self, C=10):
        self.C = C
        self._debug_minimize_info = None
        self._alphas = None
        self._support_vectors_indexes = None
        self.support_vectors = None
        self.w = None
        self.b = None

    def _lagrange_func(self, X, y, alphas):
        return - (alphas.sum() - 0.5 * np.sum([alphas[i] * alphas[j] * y[i] * y[j] * np.dot(x_i, x_j)
                                               for i, x_i in enumerate(X) for j, x_j in enumerate(X)]))

    def _cons(self, y):
        return [{"type": "eq", "fun": lambda alphas, y=y: (alphas * y).sum()},
                {"type": "ineq", "fun": lambda alphas: self.C - alphas},
                {"type": "ineq", "fun": lambda alphas: alphas}]

    def fit(self, X, y):
        y = np.copy(y)
        y[y == 0] = -1
        init_alphas = np.zeros((X.shape[0]))
        lagrange_func = partial(self._lagrange_func, X, y)
        self._debug_minimize_info = minimize(lagrange_func, init_alphas, constraints=self._cons(y))
        self._alphas = self._debug_minimize_info['x']
        self._support_vectors_indexes = np.where(self._alphas > 0.)[0]
        self.support_vectors = X[self._support_vectors_indexes, :]
        self.w = (y.reshape(-1, 1) * self._alphas.reshape(-1, 1) * X).sum(axis=0)
        self.b = ((self.w * self.support_vectors).sum(1) - y[self._support_vectors_indexes]).mean()

        return self

    def predict(self, X):
        pred = np.sign((self.w * X).sum(1) - self.b)
        pred[np.isin(pred, -1)] = 0
        return pred
