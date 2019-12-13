# You code here
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
data = load_iris()
X, y = data['data'], data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

dtree = DecisionTree()
dtree.fit(X_train, y_train)
print(dtree.predict_proba(X))
dtree_sk = DecisionTreeClassifier()
dtree_sk.fit(X_train, y_train)



import pandas as pd

data = pd.read_csv('task2.csv')
X = data.iloc[:, [0, 1]]
y = data.iloc[:, -1]
tree = DecisionTreeClassifier(max_depth=4,)
ada = AdaBoostClassifier(estimator=tree, n_estimators=200)
ada.fit(X, y)
print((ada.predict(X) == y).sum())


from xgboost import XGBClassifier