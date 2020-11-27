import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from adaboost import AdaBoostClassifierCustom
from test.test_template import TestCase


class TestAdaBoost(TestCase):
    def test_adaboost_classification(self):
        data = load_breast_cancer()
        custom_ada = AdaBoostClassifierCustom(n_estimators=20, max_depth=3)
        sklearn_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=20, random_state=10)
        self._test_template(data, accuracy_score, custom_ada, sklearn_ada, delta=0.02)


if __name__ == '__main__':
    unittest.main()
