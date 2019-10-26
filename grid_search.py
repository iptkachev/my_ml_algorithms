import numpy as np
from itertools import product
from collections import defaultdict


class MyGridSearchCV:

    def __init__(self, estimator, grid_params, kfolds, metric):
        self.estimator = estimator
        self.kfolds = kfolds
        self.metric = metric
        self.grid_params = grid_params
        self._prod_params = map(lambda t: dict(zip(grid_params.keys(), t)),
                                product(*grid_params.values()))
        self.cv_results_ = dict()
        self.best_index_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.best_estimator_ = None

    def _gen_train_test(self, X):
        indexes = np.arange(X.shape[0])
        np.random.shuffle(indexes)
        size_fold = int(X.shape[0] / self.kfolds)
        start_index, end_index = 0, size_fold
        for _ in range(self.kfolds):
            yield np.isin(indexes, indexes[start_index:end_index])
            start_index += size_fold
            end_index += size_fold

    def cross_val(self, X, y, params, i):
        self.cv_results_[i]['params'].append(params)
        for test_index in self._gen_train_test(X):
            self.estimator = self.estimator.set_params(**params)
            self.estimator.fit(X[~test_index, :], y[~test_index])
            test_metric = self.metric(y[test_index].astype('int'),
                                      self.estimator.predict(X[test_index, :]).astype('int'))
            self.cv_results_[i]['test_score'].append(test_metric)

        self.cv_results_[i]['mean_score'].append(np.array(self.cv_results_[i]['test_score']).mean())

    def fit(self, X, y):
        for i, params in enumerate(self._prod_params):
            print(i, params)
            self.cv_results_[i] = defaultdict(list)
            self.cross_val(X, y, params, i)
        len_prod_prams = i
        self.best_index_ = np.argmax([self.cv_results_[i]['mean_score'] for i in range(len_prod_prams + 1)])
        self.best_score_ = self.cv_results_[self.best_index_]['mean_score'][0]
        self.best_params_ = self.cv_results_[self.best_index_]['params'][0]
        self.best_estimator_ = self.estimator.set_params(**self.best_params_)
