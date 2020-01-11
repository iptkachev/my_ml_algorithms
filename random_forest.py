import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from decision_tree import DecisionTree


class RandomForestClassifierCustom(BaseEstimator):
    def __init__(self, n_estimators=10, max_features=0.6, max_depth=None, random_state=17):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.feat_ids_by_tree = []

    def _isinstance_check(self, array):
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        return array

    def fit(self, X, y):
        X = self._isinstance_check(X)
        y = self._isinstance_check(y)
        size_X = X.shape[0]
        indexes = np.arange(0, size_X)
        col_indexes = np.arange(0, X.shape[1])
        if X.shape[1] >= int(X.shape[1] * self.max_features):
            n_feat = X.shape[1]
        else:
            n_feat = int(X.shape[1] * self.max_features)

        for i in range(self.n_estimators):
            np.random.seed(self.random_state + i)
            self.feat_ids_by_tree.append(np.random.choice(col_indexes, size=n_feat, replace=False))
            bootstrap_samples = np.random.choice(indexes, size=size_X, replace=True)
            X_b = X[bootstrap_samples, :][:, self.feat_ids_by_tree[-1]]
            y_b = y[bootstrap_samples]
            dt = DecisionTreeClassifier(criterion='gini', max_depth=self.max_depth, random_state=self.random_state)
            dt.fit(X_b, y_b)
            self.trees.append(dt)

        return self

    def predict_proba(self, X):
        X = self._isinstance_check(X)
        predict_pr = 0.
        for i, tree in enumerate(self.trees):
            predict_pr += tree.predict_proba(X)
        predict_pr /= self.n_estimators

        return predict_pr

    def predict(self, X):
        # TODO two colums for predict proba
        return (self.predict_proba(X) > 0.5).astype(int)[:, 1]
