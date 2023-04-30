import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

boston = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
boston.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = boston.iloc[:, :-1]
y = boston.iloc[:, -1]
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)

predictions = lr.predict(X_test)
plt.scatter(y_test, predictions, color="blue")
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.show()
