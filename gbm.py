import numpy as np
from tqdm import tqdm
from functools import partial
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss, mean_squared_error
from decision_tree import DecisionTree
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingCustom(BaseEstimator):
    def __init__(self,  loss, use_custom_tree=False, n_estimators=10, learning_rate=1e-1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees_ = []
        self.loss_by_iter_ = []
        self.residuals_by_iter_ = []

        if use_custom_tree:
            self.tree_class = partial(DecisionTree, criterion='variance', max_depth=self.max_depth)
        else:
            self.tree_class = partial(DecisionTreeRegressor, max_depth=self.max_depth)

        if loss == 'mse':
            self.loss = mean_squared_error
            self.grad = self.mse_grad

        elif loss == 'rmsle':
            self.loss = self.rmsle
            self.grad = self.rmsle_grad

        elif loss == 'log_loss':
            self.loss = self.log_loss
            self.grad = self.log_loss_grad

        elif loss == 'poisson':
            self.loss = self.poisson
            self.grad = self.grad_possion

    def _sigma(self, z):
        z[z > 100] = 100
        z[z < -100] = -100
        return 1 / (1 + np.exp(-z))

    def log_loss(self, y, p):
        return log_loss(y, p)

    def _log_loss_grad(self, y, p):
        p[p == 0.] = 10e-5
        p[p == 1.] = 1 - 10e-5
        return (p - y) / (p * (1 - p))

    def _mse_grad(self, y, p):
        return - 2 / y.size * (y - p)

    def rmsle(self, y, p):
        return np.sqrt(np.power(np.log1p(y) - np.log1p(p), 2).mean())

    def _rmsle_grad(self, y, p):
        p[p < 10e-5] = 10e-5
        return np.power(p.size * (p + 1) * self.rmsle(y, p), -1) * np.log((p + 1) / (y + 1))

    def poisson(self, y, p):
        return (p - y * np.log1p(p)).mean()

    def _grad_possion(self, y, p):
        return - y / ((1 + p) * y.size)

    def fit(self, X, y):
        est = self.tree_class()
        est.fit(X, y)
        y_t = est.predict(X)

        for _ in tqdm(range(self.n_estimators)):
            self.trees_.append(est)
            loss_t = self.loss(y, y_t)
            self.loss_by_iter_.append(loss_t)
            residuals_t = - self.grad(y, y_t)
            est = self.tree_class()
            est.fit(X, residuals_t)
            h_t = est.predict(X)
            y_t += self.learning_rate * h_t
        return self

    def predict_proba(self, X):
        y_pred = self.trees_[0].predict(X)
        for tree in tqdm(self.trees_[1:]):
            y_pred += tree.predict(X)
        return self.sigma(y_pred)

    def predict(self, X):
        y_pred = self.trees_[0].predict(X)
        for tree in tqdm(self.trees_[1:]):
            y_pred += tree.predict(X)
        if self.loss in [self.log_loss]:
            return (self.sigma(y_pred) > 0.5).astype(int)
        else:
            return y_pred
