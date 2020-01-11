import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from adaboost import AdaBoostClassifier
from random_forest import RandomForestClassifierCustom
from bagging_classifier import BaggingClassifierCustom
from gbm import GradientBoostingCustom

n_est = 2
random_state = 18
gbm = GradientBoostingCustom(loss='log_loss', n_estimators=n_est, random_state=random_state)
bag = BaggingClassifierCustom(DecisionTreeClassifier(criterion='gini'), f1_score,
                              n_estimators=n_est, random_state=random_state)
ada = AdaBoostClassifier(n_estimators=n_est, random_state=random_state)
rf = RandomForestClassifierCustom(n_estimators=n_est, random_state=random_state)


df = pd.read_csv('./data/credit_scoring_sample.csv', sep=';').dropna()
y = df['SeriousDlqin2yrs'].fillna(0)
X = df.drop('SeriousDlqin2yrs', axis=1)

rf.fit(X, y)
print('rf')
bag.fit(X, y)
print('bag')
ada.fit(X, y)
print('ada')
gbm.fit(X, y)
print('gbm')

gbm_pred = gbm.predict(X)
rf_pred = rf.predict(X)
ada_pred = ada.predict(X)
bag_pred = bag.predict(X)
print(y)
print(np.unique(gbm_pred))
print(f1_score(y, gbm_pred))
print(f1_score(y, rf_pred))
print(f1_score(y, ada_pred))
print(f1_score(y, bag_pred))
