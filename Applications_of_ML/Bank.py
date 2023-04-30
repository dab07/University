import bank as bank
import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


from sklearn.metrics import accuracy_score


bank['default'] = bank.default.map({'yes':1, 'no':0})
bank['housing'] = bank.housing.map({'yes':1, 'no':0})
bank['loan'] = bank.loan.map({'yes':1, 'no':0})
bank['deposit'] = bank.deposit.map({'yes':1, 'no':0})
for col in['job', 'marital', 'education', 'contact', 'month', 'poutcome']:
print(bank.head())


bank = pd.concat([bank.drop(col, axis = 1),pd.get_dummies(bank[col], prefix=col, prefix_sep='_', drop_first = True, dummy_na = False)], axis=1)
bank.drop('contact_unknown', axis=1, inplace=True)
X = bank.drop('deposit', axis=1)
y = bank['deposit']
bank.drop('pdays', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 100)

xgb = xgboost.XGBClassifier(n_estimators = 100, learning_rate = 0.08, gamma = 0, subsample = 0.75, colsample_bytree = 1, max_depth = 7)
xgb.fit(X_train, y_train.squeeze().values)
y_train_preds = xgb.predict(X_train)
y_test_preds = xgb.predict(X_test)


print('XGB accuracy for training: %.3f')
accuracy_score(y_train, y_train_preds)
print('XGB accuracy score for Test: %.3f')
accuracy_score(y_test, y_test_preds)


headers = ['name', 'score']
values = sorted(zip(X_train.columns, xgb.feature_importances_), key=lambda x:x[1]*-1)
xgb_feature_importances = pd.DataFrame(values, columns = headers)
xgb_feature_importances


import matplotlib.pyplot as plt
x_pos = np.arange(0, len(xgb_feature_importances))
plt.bar(x_pos, xgb_feature_importances['score'])
plt.xticks(x_pos, xgb_feature_importances['name'])
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.show()