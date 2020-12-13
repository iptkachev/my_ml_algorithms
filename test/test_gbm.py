import unittest
from gbm import GradientBoostingCustom
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from test.test_template import TestCase


class TestGradientBoostingCustom(TestCase):
    def test_custom_gbm_classification(self):
        data = load_breast_cancer()
        custom_gbm = GradientBoostingCustom('log_loss', use_custom_tree=True, n_estimators=10,
                                            max_depth=3, learning_rate=0.1)
        sklearn_gbm = GradientBoostingClassifier('deviance', learning_rate=0.1, n_estimators=10, max_depth=3)
        self._test_template(data, accuracy_score, custom_gbm, sklearn_gbm, delta=0.05)

    def test_custom_gbm_regression(self):
        data = load_diabetes()
        custom_gbm = GradientBoostingCustom('mse', use_custom_tree=True, n_estimators=10,
                                            max_depth=3, learning_rate=0.1)
        sklearn_gbm = GradientBoostingRegressor('ls', learning_rate=0.1, n_estimators=10, max_depth=3)
        self._test_template(data, mean_squared_error, custom_gbm, sklearn_gbm, delta=400)


if __name__ == '__main__':
    unittest.main()
