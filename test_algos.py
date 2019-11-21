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