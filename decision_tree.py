import numpy as np
import warnings
from collections import Counter
from sklearn.base import BaseEstimator
from tree_criterions import *
warnings.filterwarnings('ignore')


class Node:
    def __init__(self, feature_idx=None, threshold=None, depth=None, labels=None, left=None, right=None):
        self.depth = depth
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right


class DecisionTree(BaseEstimator):
    def __init__(self, max_depth=np.inf, min_samples_split=2, criterion='gini', random_state=17):
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.nodes = []

        if self.criterion == 'gini':
            self.func_criterion = gini
            self.classes_indexes = None
        elif self.criterion == 'entropy':
            self.func_criterion = entropy
            self.classes_indexes = None
        elif self.criterion == 'variance':
            self.func_criterion = variance
        elif self.criterion == 'mad_median':
            self.func_criterion = mad_median

    def _isinstance_check(self, array):
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        return array

    def _cond(self, node, y):
        return np.unique(y[node.labels]).shape[0] != 1 and \
               node.depth < self.max_depth and \
               node.labels.size >= self.min_samples_split

    def make_split(self, features, X, y, parent_node):
        labels = parent_node.labels
        n = labels.shape[0]
        best_feat_idx = None
        best_t = None
        best_ig = 0.

        for i_feat in features:
            X_feat = X[labels, i_feat]
            for t in np.linspace(X_feat.min(), X_feat.max())[1:-1]:
                left_ind = labels[np.where(X_feat < t)[0]]
                n_left = left_ind.size
                right_ind = labels[np.where(X_feat >= t)[0]]
                n_rigth = right_ind.size
                info_gain = self.func_criterion(y[labels]) - n_left / n * self.func_criterion(y[left_ind]) \
                            - n_rigth / n * self.func_criterion(y[right_ind])

                if info_gain > best_ig:
                    best_t = t
                    best_feat_idx = i_feat
                    best_ig = info_gain

        parent_node.feature_idx = best_feat_idx
        parent_node.threshold = best_t
        left_ind = labels[np.where(X[labels, best_feat_idx] < best_t)[0]]
        right_ind = labels[np.where(X[labels, best_feat_idx] >= best_t)[0]]
        left = Node(depth=parent_node.depth + 1, labels=left_ind)
        right = Node(depth=parent_node.depth + 1, labels=right_ind)
        parent_node.left = left
        parent_node.right = right

        return left, right

    def fit(self, X, y):
        features = range(X.shape[1])
        X = self._isinstance_check(X)
        y = self._isinstance_check(y)
        if self.criterion in ['gini', 'entropy']:
            self.classes_indexes = dict(zip(np.unique(y), range(np.unique(y).size)))
        root = Node(depth=0, labels=np.arange(y.shape[0]))
        self.nodes.append(root)

        for node in self.nodes:
            if self._cond(node, y):
                left, right = self.make_split(features, X, y, node)
                self.nodes.extend([left, right])
            else:
                node.labels = y[node.labels]

        return self

    def _down_to_leaf(self, obj, node):
        while node.left and node.right:
            if obj[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, obj in enumerate(X):
            root = self.nodes[0]
            leaf = self._down_to_leaf(obj, root)
            if self.criterion in ['gini', 'entropy']:
                y_pred_obj = Counter(leaf.labels).most_common(1)[0][0]
            elif self.criterion in ['variance', 'mad_median']:
                y_pred_obj = leaf.labels.mean()
            y_pred[i] = y_pred_obj

        return y_pred

    def predict_proba(self, X):
        if self.criterion in ['gini', 'entropy']:
            y_pred = np.zeros((X.shape[0], len(self.classes_indexes)))
            for i, obj in enumerate(X):
                root = self.nodes[0]
                leaf = self._down_to_leaf(obj, root)
                y_obj_unique, y_obj_counts = np.unique(leaf.labels, return_counts=True)
                probs = y_obj_counts / y_obj_counts.sum()
                for i_un_cl, un_cl in enumerate(y_obj_unique):
                    y_pred[i, self.classes_indexes[un_cl]] = probs[i_un_cl]
            return y_pred
        else:
            raise ValueError
