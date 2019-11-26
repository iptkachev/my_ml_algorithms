import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier


class AdaBoostClassifier(BaseEstimator):
    def __init__(self, estimator, n_estimators=10, random_state=17, verbose=50):
        self.class_estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.verbose = verbose
        self.estimators = []
        self.lr_t = []

    def _isinstance_check(self, array):
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        return array

    def fit(self, X, y):
        X = self._isinstance_check(X)
        y = self._isinstance_check(y)
        if 0 in np.unique(y):
            y[y == 0] = -1

        sample_weights = np.ones(y.shape[0]) / y.shape[0]
        for i in range(self.n_estimators):
            estimator = self.class_estimator
            estimator.fit(X, y, sample_weight=sample_weights)
            erros = (sample_weights * (estimator.predict(X) != y)).sum()
            if erros >= 0.5 or erros == 0:
                print(f'early stop with {i + 1} estimators')
                break
            elif i % self.verbose  == 0:
                print('errors sum: ', erros)
            lr_t = 1 / 2 * np.log((1 - erros) / erros)
            self.estimators.append(estimator)
            self.lr_t.append(lr_t)
            sample_weights *= np.exp(- lr_t * y * estimator.predict(X))
            sample_weights = sample_weights / sample_weights.sum()

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for i, est in enumerate(self.estimators):
            preds += self.lr_t[i] * est.predict(X)

        return np.sign(preds)


import pandas as pd

data = pd.read_csv('task2.csv')
X = data.iloc[:, [0, 1]]
y = data.iloc[:, -1]
tree = DecisionTreeClassifier(max_depth=4,)
ada = AdaBoostClassifier(estimator=tree, n_estimators=200)
ada.fit(X, y)
print((ada.predict(X) == y).sum())
