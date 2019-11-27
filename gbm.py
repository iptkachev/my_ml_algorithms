import numpy as np
from tqdm import tqdm_notebook
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, accuracy_score
from decision_tree import DecisionTree


class GradientBoosting(BaseEstimator):
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
            self.estimator = DecisionTree(criterion='variance', max_depth=self.max_depth,
                                          random_state=self.random_state)
        elif loss == 'log_loss':
            self.loss = log_loss
            self.grad = self.log_loss_grad
            self.estimator = DecisionTree(criterion='gini', max_depth=self.max_depth,
                                          random_state=self.random_state)
        elif loss == 'rmsle':
            self.loss = self.rmsle
            self.grad = self.rmsle_grad
            self.estimator = DecisionTree(criterion='variance', max_depth=self.max_depth,
                                          random_state=self.random_state)

    def sigma(self, z):
        z[z > 100] = 100
        z[z < -100] = -100
        return 1 / (1 + np.exp(-z))

    def log_loss(self, y, p):
        return log_loss(y, p)

    def log_loss_grad(self, y, p):
        return (p - y) / (p * (1 - p))

    def mse_grad(self, y, p):
        return 2 / y.size * (y - p)

    def rmsle(self, y, p):
        return np.sqrt(np.power(np.log1p(y) - np.log1p(p), 2).mean())

    def rmsle_grad(self, y, p):
        return np.power(p.size * (p + 1) * self.rmsle(y, p), -1) * np.log((p + 1) / (y + 1))

    def fit(self, X, y):
        self.estimator.fit(X, y)
        t_loss = self.loss(y, self.estimator.predict(X))
        t_residuals = self.grad(y, self.estimator.predict(X))
        self.loss_by_iter_.append(t_loss)
        self.trees_.append(self.estimator)
        if self.debug:
            self.residuals_by_iter_.append(t_residuals)
        y_approx = y + t_residuals

        for t in tqdm_notebook(range(self.n_estimators)):
            print('iter', t)
            self.estimator.fit(X, y_approx)
            t_loss = self.loss(y_approx, self.estimator.predict(X))
            t_residuals = self.grad(y_approx, self.estimator.predict(X))
            self.loss_by_iter_.append(t_loss)
            self.trees_.append(self.estimator)
            if self.debug:
                self.residuals_by_iter_.append(t_residuals)
            y_approx += t_residuals

        return self

    def predict_proba(self, X):
        if self.loss in [log_loss]:
            y_pred = np.zeros(X.shape[0])
            for tree in self.trees_:
                y_pred += tree.predict(X)
            return self.sigma(y_pred)
        else:
            raise ValueError

    def predict(self, X):
        if self.loss in [log_loss]:
            return (self.predict_proba(X) > 0.5).astype(int)
        else:
            y_pred = np.zeros(X.shape[0])
            for tree in self.trees_:
                y_pred += tree.predict(X)
            return y_pred