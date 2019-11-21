import numpy as np
from sklearn.base import BaseEstimator
from decision_tree import DecisionTree


class BaggingClassifierCustom(BaseEstimator):
    def __init__(self, metric, n_estimators=10, max_depth=3, random_state=17):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.metric = metric
        # в данном списке будем хранить отдельные деревья
        self.trees = []
        # в данном списке будем хранить объекты, которые не попали в обучающую выборку
        self.oob = []

    def _isinstance_check(self, array):
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        return array

    def fit(self, X, y):
        X = self._isinstance_check(X)
        y = self._isinstance_check(y)
        indexes = np.arange(0, X.shape[0])
        size_X = X.shape[0]
        # Ваш код здесь
        for i in range(self.n_estimators):
            np.random.seed(self.random_state + i)
            bootstrap_samples = np.random.choice(indexes, size=size_X, replace=True)
            oob_indexes = indexes[~np.isin(indexes, bootstrap_samples)]

            dt = DecisionTree(max_depth=self.max_depth, random_state=self.random_state)
            dt.fit(X[bootstrap_samples, :],
                   y[bootstrap_samples])
            dt.oob_indexes = oob_indexes
            self.trees.append(dt)

        return self

    def predict_proba(self, X):

        # Ваш код здесь
        probs = np.zeros(X.shape[0])
        for i, tree in enumerate(self.trees):
            if i == 0:
                probs = tree.predict_proba(X)
            else:
                probs += tree.predict_proba(X)

        probs /= len(self.trees)

        return probs

    def oob_score(self, X, y):
        X = self._isinstance_check(X)
        y = self._isinstance_check(y)
        metric_score = 0.

        for tree in self.trees:
            metric_score += self.metric(y[tree.oob_indexes], tree.predict_proba(X[tree.oob_indexes, :])[:, 0])
        metric_score /= len(self.trees)

        return metric_score
