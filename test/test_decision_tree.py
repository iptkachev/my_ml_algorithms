import unittest
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from decision_tree import DecisionTree
from test.test_template import TestCase


class TestDecisionTree(TestCase):
    def test_dtree_classification(self):
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            data['data'], data['target'], test_size=0.2, random_state=17
        )

        data = load_iris()
        custom_dtree = DecisionTree()
        sklearn_dtree = DecisionTreeClassifier()
        self._test_template(data, accuracy_score, custom_dtree, sklearn_dtree, delta=0.07)

    def test_dtree_regression(self):
        data = load_diabetes()
        custom_dtree = DecisionTree(criterion='variance')
        sklearn_dtree = DecisionTreeRegressor()
        self._test_template(data, mean_squared_error, custom_dtree, sklearn_dtree, delta=800)


if __name__ == '__main__':
    unittest.main()
