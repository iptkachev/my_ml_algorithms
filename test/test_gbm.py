import pandas as pd
from gbm import GradientBoostingCustom
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.datasets import load_digits, load_iris, load_boston
from decision_tree import DecisionTree
# data = load_boston()
# X, y = data['data'], data['target']
# data = pd.read_csv('./data/task2.csv')
# X, y = data.iloc[:, [0, 1]], data.iloc[:, 2]
# y[y == 1] = 0
# y[y == -1] = 1
df = pd.read_csv('./data/credit_scoring_sample.csv', sep=';', dtype=float).dropna()
y = df['SeriousDlqin2yrs'].fillna(0)
X = df.drop('SeriousDlqin2yrs', axis=1)
gbm = GradientBoostingCustom('log_loss', max_depth=4, learning_rate=0.001, debug=True, n_estimators=10)
# gbm = DecisionTree(criterion='variance')
gbm.fit(X, y)
print(gbm.predict(X))
print(accuracy_score(y, gbm.predict(X)))

# print(mean_squared_error(y, gbm.predict(X)))
# print(gbm.loss_by_iter_)