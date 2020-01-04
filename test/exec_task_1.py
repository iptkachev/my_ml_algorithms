import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from custom_nn.neural_network import Net, Layer
from custom_nn.utils import LogLoss, Sigmoid


df = pd.read_csv('task1.csv')

X = df.iloc[:, [0, 1]]
y = df.iloc[:, [2]]
y[y == -1] = 0

net = Net(LogLoss(), random_state=10)
net.add_layer(Layer(input_size=2, output_size=2, activate_fn=Sigmoid()))
net.add_layer(Layer(input_size=2, output_size=1, activate_fn=Sigmoid(), last_layer=True))
net.train(X.values, y.values, learning_rate=0.15, epochs=450, batch_size=128, verbose=200)
pred = net(X.values)
pred = (pred > 0.5).astype('int')
print(np.unique(pred))
print(accuracy_score(y, pred))