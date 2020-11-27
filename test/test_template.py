import numpy as np
import unittest
from typing import Dict, Callable
from sklearn.model_selection import train_test_split


class TestCase(unittest.TestCase):
    def _test_template(self, data: Dict[str, np.array], metric: Callable, custom_algorithm, sklearn_algorithm,
                       **assert_kwargs):
        X_train, X_test, y_train, y_test = train_test_split(
            data['data'], data['target'], test_size=0.2, random_state=17
        )

        custom_algorithm = custom_algorithm
        custom_algorithm.fit(X_train, y_train)
        custom_train_predict = custom_algorithm.predict(X_train)
        custom_test_predict = custom_algorithm.predict(X_test)
    
        sklearn_algorithm = sklearn_algorithm
        sklearn_algorithm.fit(X_train, y_train)
        sklearn_train_predict = sklearn_algorithm.predict(X_train)
        sklearn_test_predict = sklearn_algorithm.predict(X_test)
    
        sklearn_train_metric = metric(y_train, sklearn_train_predict)
        custom_train_metric = metric(y_train, custom_train_predict)
        self.assertAlmostEqual(sklearn_train_metric, custom_train_metric, **assert_kwargs)
    
        sklearn_test_metric = metric(y_test, sklearn_test_predict)
        custom_test_metric = metric(y_test, custom_test_predict)
        self.assertAlmostEqual(sklearn_test_metric, custom_test_metric, **assert_kwargs)
