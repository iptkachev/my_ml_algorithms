import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier


class AdaBoostClassifier(BaseEstimator):
    def __init__(self, n_estimators=10, random_state=17, verbose=50):
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
        X = np.copy(self._isinstance_check(X))
        y = np.copy(self._isinstance_check(y))
        if 0 in np.unique(y):
            y[y == 0] = -1

        sample_weights = np.ones(y.shape[0]) / y.shape[0]
        for i in range(self.n_estimators):
            estimator = DecisionTreeClassifier()
            estimator.fit(X, y, sample_weight=sample_weights)
            erros = (sample_weights * (estimator.predict(X) != y)).sum()
            if erros >= 0.5 or erros == 0:
                print(f'early stop with {i + 1} estimators')
                break
            elif i % self.verbose == 0:
                print('errors sum: ', erros)
            lr_t = 1 / 2 * np.log((1 - erros) / erros)
            self.estimators.append(estimator)
            self.lr_t.append(lr_t)
            sample_weights *= np.exp(- lr_t * y * estimator.predict(X))
            sample_weights = sample_weights / sample_weights.sum()

        return self

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for i, est in enumerate(self.estimators):
            preds += self.lr_t[i] * est.predict(X)
        preds[preds <= 0] = 0
        preds[preds > 0] = 1
        return preds



