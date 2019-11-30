import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from decision_tree import DecisionTree


class GradientBoostingCustom(BaseEstimator):
    def __init__(self, loss='mse', n_estimators=10, learning_rate=10e-2, max_depth=3,
                 debug=False, random_state=17):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.debug = debug
        self.trees_ = []
        self.loss_by_iter_ = []
        self.residuals_by_iter_ = []

        if loss == 'mse':
            self.loss = mean_squared_error
            self.grad = self.mse_grad

        elif loss == 'rmsle':
            self.loss = self.rmsle
            self.grad = self.rmsle_grad

        elif loss == 'log_loss':
            self.loss = log_loss
            self.grad = self.log_loss_grad

        elif loss == 'poisson':
            self.loss = self.poisson
            self.grad = self.grad_possion

    def sigma(self, z):
        z[z > 100] = 100
        z[z < -100] = -100
        return 1 / (1 + np.exp(-z))

    def log_loss(self, y, p):
        return log_loss(y, p)

    def log_loss_grad(self, y, p):
        p[p < 10e-5] = 10e-5
        return (p - y) / (p * (1 - p))

    def mse_grad(self, y, p):
        return - 2 / y.size * (y - p)

    def rmsle(self, y, p):
        return np.sqrt(np.power(np.log1p(y) - np.log1p(p), 2).mean())

    def rmsle_grad(self, y, p):
        p[p < 10e-5] = 10e-5
        return np.power(p.size * (p + 1) * self.rmsle(y, p), -1) * np.log((p + 1) / (y + 1))

    def poisson(self, y, p):
        return - (y * p - np.exp(p)).sum()

    def grad_possion(self, y, p):
        return - y + np.exp(p)

    def fit(self, X, y):
        if self.loss in [mean_squared_error, self.rmsle]:
            est = DecisionTreeRegressor(max_depth=self.max_depth)
        elif self.loss in [self.poisson, log_loss]:
            est = DecisionTreeClassifier(max_depth=self.max_depth)
            est.predict = est.predict_proba
        est.fit(X, y)
        y_t = est.predict(X)
        if len(y_t.shape) == 2:
            y_t = y_t[:, 0].reshape(-1)
        self.trees_.append(est)
a
        for _ in tqdm(range(self.n_estimators)):
            loss_t = self.loss(y, y_t)
            residuals_t = - self.grad(y, y_t)
            if self.loss in [mean_squared_error, self.rmsle]:
                est = DecisionTreeRegressor(max_depth=self.max_depth)
            elif self.loss in [self.poisson, log_loss]:
                est = DecisionTreeRegressor(max_depth=self.max_depth)
            est.fit(X, residuals_t)
            self.loss_by_iter_.append(loss_t)
            self.trees_.append(est)
            if self.debug:
                self.residuals_by_iter_.append(residuals_t)
            h_t = est.predict(X)
            if len(y_t.shape) == 2:
                h_t = h_t[:, 0].reshape(-1)
            y_t += self.learning_rate * h_t

        return self

    def predict_proba(self, X):
        if self.loss in [log_loss]:
            y_pred = self.trees_[0].predict(X)
            for est in self.trees_[1:]:
                y_pred += self.learning_rate * est.predict(X)
            return self.sigma(y_pred)
        else:
            raise ValueError

    def predict(self, X):
        if self.loss in [log_loss]:
            return (self.predict_proba(X) > 0.5).astype(int)
        else:
            y_pred = self.trees_[0].predict(X)
            print(y_pred)
            for tree in self.trees_[1:]:
                y_pred += self.learning_rate * tree.predict(X)
            return y_pred

import pandas as pd
from sklearn.datasets import load_digits, load_iris, load_boston
data = load_boston()
X, y = data['data'], data['target']
data = pd.read_csv('task2.csv')
X, y = data.iloc[:, [0, 1]], data.iloc[:, 2]
gbm = GradientBoostingCustom('log_loss', max_depth=2, learning_rate=0.1, debug=True)
gbm.fit(X, y)
print(gbm.predict(X))
print(log_loss(y, gbm.predict(X)))
# print(mean_squared_error(y, gbm.predict(X)))
print(gbm.loss_by_iter_)