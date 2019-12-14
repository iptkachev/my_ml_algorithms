import pandas as pd
from cnn.neural_network import Net, Layer
from cnn.utils import LogLoss
df = pd.read_csv('task2.csv')
X = df.iloc[:, [0, 1]]
y = df.iloc[:, [2]]
y[y == -1] = 0
# layer
# layer = Layer(X.shape[1], 1, 'sigmoid')
# print(layer.forward(X))
# print(layer.backward(X))

# net
net = Net(LogLoss, Layer(input_size=2, output_size=3, activate_fn='relu'))
net.train(X.values, y.values, 0.05, 1, 8)
