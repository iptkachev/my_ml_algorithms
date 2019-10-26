import numpy as np
from functools import partial
from scipy.optimize import minimize
from sklearn.base import BaseEstimator


class MySVC(BaseEstimator):

    def __init__(self, C=10):
        self.C = C
        self._info = None
        self._lagrange_f = None
        self.support_vectors = None
        self.w = None
        self.b = None

    @classmethod
    def _lagrange_func(self, X, y, alphas):
        return - (alphas.sum() - 0.5 * np.sum([alphas[i] * alphas[j] * y[i] * y[j] * np.dot(x_i, x_j)
                                               for i, x_i in enumerate(X) for j, x_j in enumerate(X)]))

    def _cons(self, y):
        return [{"type": "eq", "fun": lambda alphas, y=y: (alphas * y).sum()},
                {"type": "ineq", "fun": lambda alphas: self.C - alphas},
                {"type": "ineq", "fun": lambda alphas: alphas}]

    def fit(self, X, y):
        y[y == 0] = -1
        init_alphas = np.zeros((X.shape[0]))
        self._lagrange_f = partial(self._lagrange_func, X, y)
        self._info = minimize(self._lagrange_f, init_alphas, constraints=self._cons(y))
        alphas = self._info['x']
        index_sv = np.where(alphas > 1e-04)[0]
        self.support_vectors = X[index_sv, :]
        self.w = (y.reshape(-1, 1) * alphas.reshape(-1, 1) * X).sum(axis=0)
        self.b = (self.w * X[index_sv[0], :]).sum() - y[index_sv[0]]

    def predict(self, X):
        pred = np.sign((self.w * X).sum(1) - self.b)
        pred[np.isin(pred, (-1, 0))] = 0
        return pred
