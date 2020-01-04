import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,mean_squared_error
from custom_nn.neural_network import Net, Layer
from custom_nn.utils import LogLoss, ReLU, Sigmoid, MSELoss, Linear
sc = StandardScaler()
data = load_boston()
X_r, y_r = data['data'], data['target']
X_r = sc.fit_transform(X_r)

# df = pd.read_csv('task2.csv')
# X = df.iloc[:, [0, 1]]
# y = df.iloc[:, [2]]
# y[y == -1] = 0

df = pd.read_csv("./apples_pears.csv")
X = df.iloc[:, [0, 1]]
y = df.iloc[:, [2]]

# layer
# layer = Layer(X.shape[1], 1, 'sigmoid')
# print(layer.forward(X))
# print(layer.backward(X))

# net classification
net = Net(LogLoss, random_state=10)
net.add_layer(Layer(input_size=2, output_size=20, activate_fn=Linear()))
net.add_layer(Layer(input_size=20, output_size=2, activate_fn=Sigmoid()))
net.add_layer(Layer(input_size=2, output_size=1, activate_fn=Sigmoid(), last_layer=True))
net.train(X.values, y.values, 0.5, 1000, 128)
pred = net(X.values)
pred = (pred > 0.5).astype('int')
print(np.unique(pred))
print(accuracy_score(y.values.reshape(-1), pred))

# net regressian
# net = Net(MSELoss(), random_state=10)
# net.add_layer(Layer(input_size=13, output_size=10, activate_fn=Linear()))
# net.add_layer(Layer(input_size=10, output_size=10, activate_fn=Sigmoid()))
# net.add_layer(Layer(input_size=10, output_size=1, activate_fn=Linear()))
# net.forward(X_r)
# net.train(X_r, y_r.reshape(-1, 1), 0.01, 1000, 32)
# pred = net(X_r)
