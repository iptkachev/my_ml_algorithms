import pandas as pd
import numpy as np
from gbm import GradientBoostingCustom
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.datasets import load_digits, load_iris, load_boston, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from decision_tree import DecisionTree
# data = load_boston()
# X, y = data['data'], data['target']
# gbm = GradientBoostingCustom('mse', max_depth=3, learning_rate=0.1, debug=True, n_estimators=20)
# gbm.fit(X, y)
# print(mean_squared_error(y, gbm.predict(X)))
# print(mean_squared_error(y, np.array([y.mean()]*y.shape[0])))
# data = pd.read_csv('./data/task2.csv')
# X, y = data.iloc[:, [0, 1]], data.iloc[:, 2]
# y[y == 1] = 0
# y[y == -1] = 1
df = pd.read_csv('./data/credit_scoring_sample.csv', sep=';', dtype=float).dropna()
y = df['SeriousDlqin2yrs'].fillna(0).reset_index(drop=True)
X = df.drop('SeriousDlqin2yrs', axis=1)
data = load_breast_cancer()
X, y = data['data'], data['target']
gbm = GradientBoostingCustom('log_loss', use_custom_tree=True, max_depth=3, learning_rate=0.1)
# gbm = DecisionTreeClassifier(max_depth=3)
# gbm = DecisionTree(criterion='variance')
gbm.fit(X, y)
y_pred_proba = gbm.predict_proba(X)
y_pred = gbm.predict(X)
print(y_pred)
print(accuracy_score(y, np.zeros(y.shape[0])))

print(accuracy_score(y, y_pred))
# print(mean_squared_error(y, gbm.predict(X)))
# print(gbm.loss_by_iter_)