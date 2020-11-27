import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier


class AdaBoostClassifierCustom(BaseEstimator):
    def __init__(self, n_estimators=10, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.verbose = False
        self.trees_ = []
        self.learning_rates_ = []

    def fit(self, X, y):
        y = np.copy(y)
        y[y == 0] = -1

        sample_weights = np.ones(y.shape[0]) / y.shape[0]
        for t in tqdm(range(self.n_estimators)):
            estimator = DecisionTreeClassifier(max_depth=self.max_depth)
            estimator.fit(X, y, sample_weight=sample_weights)

            predict = estimator.predict(X)
            errors_t = (sample_weights * (predict != y)).sum() / sample_weights.sum()
            if not errors_t:
                print(f'early stop with {t + 1} estimators')
                break
            if self.verbose:
                print(f'errors sum: {errors_t}')

            learning_rate_ = 1 / 2 * np.log((1 - errors_t) / errors_t)
            self.trees_.append(estimator)
            self.learning_rates_.append(learning_rate_)
            sample_weights *= np.exp(-learning_rate_ * y * predict)
            sample_weights = sample_weights / sample_weights.sum()

        return self

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for t, est in enumerate(self.trees_):
            preds += self.learning_rates_[t] * est.predict(X)
        preds[preds <= 0] = 0
        preds[preds > 0] = 1
        return preds
