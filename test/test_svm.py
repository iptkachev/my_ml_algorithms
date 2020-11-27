# from sklearn.datasets import load_digits, load_iris, load_boston
# import numpy as np
#
# from svm import MySVC
#
# data = load_iris()
# # X, y = data['data'], data['target']
# # X = X[y < 2, :]
# # y = y[y < 2]
# X = np.array([[1, 3], [4, 2], [4, 5], [7, 3], [1, 9], [10, 5]])
# y = np.array([-1, -1, 1, 1, 1, 1])
# svm = MySVC(C=1000)
# svm.fit(X, y)
# print(svm.predict(X))
# print(svm.support_vectors)
#
# np.random.poisson(4, 5)
# # x = []; y = []
# # b1 = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).sum()
# # b0 = np.mean(y) - b1 * np.mean(x)
#
# from sklearn.linear_model import LinearRegression
# from svm import MySVC
#
# for i in range(4):
#     print('________________________________')
#     np.random.seed(i)
#     x = (np.arange(5) + np.random.randint(-10, 10, 5)).reshape(-1, 1)
#     np.random.seed(i + 1)
#     y = np.arange(5) + np.random.randint(-10, 10, 5)
#     lr = LinearRegression()
#     lr.fit(x, y)
#     print(i)
#     print('x', x)
#     print('y', y)
#     print('mean', x.mean(), y.mean())
#     print('b1', lr.coef_)
#     print('b0', lr.intercept_)
#     np.random.seed(i + 10)
#     x = (np.arange(3) + np.random.randint(-10, 10, 3)).reshape(-1, 1)
#     print('x_predict', x)
#     print('predict', lr.predict(x))
#     # svm
#     print('svm')
#     svm = MySVC(0.5)
#     np.random.seed(i + 2)
#     x = (np.arange(12) + np.random.randint(0, 6, 12)).reshape(-1, 2)
#     np.random.seed(i + 3)
#     y = np.array([0, 0, 0, 1, 1, 1])
#     svm.fit(x, y)
#     print('x', x)
#     print('y', y)
#     print('alphas', svm.alphas)
#     print('w', svm.w)
#     print('b', svm.b)
#     print('supp_vectors index', svm.index_sv)
#     np.random.seed(i + 20)
#     x = (np.arange(6) + np.random.randint(-20, 20, 6)).reshape(-1, 2)
#     print('x_predict', x)
#     print('predict', svm.predict(x))
#
#
# def grad(b0, b1, x, y):
#     for i in range(2):
#         b0 = b0 + 0.1 * 2 * (y - b0 - b1 * x)
#         b1 = b1 + 0.1 * 2 * x * (y - b0 - b1 * x)
#         print('b0', b0)
#         print('b1', b1)
#
# grad(0.1, 2, x=2, y=1)

import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from svm import CustomSVC
from test.test_template import TestCase


class TestCustomSVC(TestCase):
    def test_support_vector_machine_classification(self):
        data = load_breast_cancer()
        # with more than 200 samples `minimize` (in CustomSVC) works too long
        data['target'] = data['target'][:200]
        data['data'] = data['data'][:200]
        custom_svc = CustomSVC(C=0.5)
        sklearn_svc = SVC(C=0.5)
        self._test_template(data, accuracy_score, custom_svc, sklearn_svc, delta=0.05)


if __name__ == '__main__':
    unittest.main()
