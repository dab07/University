import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/hs/Downloads/50_Startups.csv")
x = df.iloc[:, :-2].values
y = df.iloc[:, 4].values
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 10)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

plt.plot(y_test, y_pred, color="red")
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.show()
