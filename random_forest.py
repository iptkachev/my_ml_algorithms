import numpy as np
from sklearn.base import BaseEstimator
from decision_tree import DecisionTree


class RandomForestClassifierCustom(BaseEstimator):
    def __init__(self, n_estimators=10, max_depth=10, max_features=10, random_state=17):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.feat_ids_by_tree = []

    def fit(self, X, y):
        size_X = len(X)
        indexes = np.arange(0, size_X)
        col_indexes = np.arange(0, X.shape[1])

        for i in range(self.n_estimators):
            np.random.seed(self.random_state + i)
            self.feat_ids_by_tree.append(np.random.choice(col_indexes,
                                                          size=self.max_features
                                                          if X.shape[1] >= self.max_features
                                                          else X.shape[1],
                                                          replace=False))
            bootstrap_samples = np.random.choice(indexes, size=size_X, replace=True)

            dt = DecisionTree(max_depth=self.max_depth, random_state=self.random_state)
            dt.fit(X.values[bootstrap_samples, :][:, self.feat_ids_by_tree[-1]],
                   y.values[bootstrap_samples])
            self.trees.append(dt)

        return self

    def predict_proba(self, X):
        predict_pr = 0.
        for i, tree in enumerate(self.trees):
            predict_pr += tree.predict_proba(X.values[:, :])
        predict_pr /= self.n_estimators

        return predict_pr
