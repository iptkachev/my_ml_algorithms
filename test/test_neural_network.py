import pandas as pd
import numpy as np
import unittest
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes, load_boston
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from custom_nn.neural_network import Net, Layer
from custom_nn.utils import LogLoss, MSELoss, ReLU, Sigmoid, Linear
from test.test_template import TestCase


class TestNeuralNetworkCustom(TestCase):
    def test_custom_neural_network_classification(self):
        df = pd.read_csv("./data/apples_pears.csv").values
        X = df[:, [0, 1]]
        y = df[:, [2]]
        input_size = X.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=17, shuffle=True
        )

        custom_mlp = Net(LogLoss(), random_state=10)
        custom_mlp.add_layer(Layer(input_size=input_size, output_size=20, activate_fn=Linear()))
        custom_mlp.add_layer(Layer(input_size=20, output_size=2, activate_fn=Sigmoid()))
        custom_mlp.add_layer(Layer(input_size=2, output_size=1, activate_fn=Sigmoid(), last_layer=True))

        custom_mlp.fit(X_train, y_train, learning_rate=0.001, epochs=1000, batch_size=128)
        custom_train_predict = np.array(custom_mlp(X_train).reshape(-1) >= 0.5, dtype=int)
        custom_test_predict = np.array(custom_mlp(X_test).reshape(-1) >= 0.5, dtype=int)

        sklearn_mlp = MLPClassifier((3, 20), activation='logistic', solver='sgd', random_state=10)
        sklearn_mlp.fit(X_train, y_train)
        sklearn_train_predict = sklearn_mlp.predict(X_train)
        sklearn_test_predict = sklearn_mlp.predict(X_test)

        sklearn_train_metric = accuracy_score(y_train, sklearn_train_predict)
        custom_train_metric = accuracy_score(y_train, custom_train_predict)
        self.assertTrue(sklearn_train_metric <= custom_train_metric)

        sklearn_test_metric = accuracy_score(y_test, sklearn_test_predict)
        custom_test_metric = accuracy_score(y_test, custom_test_predict)
        self.assertTrue(sklearn_test_metric <= custom_test_metric)

    def test_custom_neural_network_regression(self):
        data = load_diabetes()
        input_size = data['data'].shape[1]
        sc = StandardScaler()
        data['data'] = sc.fit_transform(data['data'])
        X_train, X_test, y_train, y_test = train_test_split(
            data['data'], data['target'], test_size=0.2, random_state=17
        )

        custom_mlp = Net(MSELoss(), random_state=10)
        custom_mlp.add_layer(Layer(input_size=input_size, output_size=10, activate_fn=Sigmoid()))
        custom_mlp.add_layer(Layer(input_size=10, output_size=10, activate_fn=Sigmoid()))
        custom_mlp.add_layer(Layer(input_size=10, output_size=1, activate_fn=Linear(), last_layer=True))

        custom_mlp.fit(X_train, y_train, learning_rate=0.02, epochs=200, batch_size=512, verbose=50)
        custom_train_predict = custom_mlp(X_train).reshape(-1)
        custom_test_predict = custom_mlp(X_test).reshape(-1)

        sklearn_mlp = MLPRegressor((3, 30), activation='logistic', solver='sgd', random_state=10)
        sklearn_mlp.fit(X_train, y_train)
        sklearn_train_predict = sklearn_mlp.predict(X_train)
        sklearn_test_predict = sklearn_mlp.predict(X_test)

        sklearn_train_metric = mean_squared_error(y_train, sklearn_train_predict)
        custom_train_metric = mean_squared_error(y_train, custom_train_predict)
        self.assertAlmostEqual(sklearn_train_metric, custom_train_metric, delta=sklearn_train_metric * 0.3)

        sklearn_test_metric = mean_squared_error(y_test, sklearn_test_predict)
        custom_test_metric = mean_squared_error(y_test, custom_test_predict)
        self.assertAlmostEqual(sklearn_test_metric, custom_test_metric, delta=sklearn_train_metric * 0.3)


if __name__ == '__main__':
    unittest.main()
