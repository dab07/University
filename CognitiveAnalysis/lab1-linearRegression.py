import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/hs/Downloads/test.csv")

x = df[["x"]]
y = df[["y"]]

print(df.describe())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)

# PREDICTING DATA
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions, color="blue")
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.show()


# EVALUATING THE MODEL
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

from sklearn.metrics import accuracy_score, r2_score
print(r2_score(y_test, predictions))
sb.displot((y_test-predictions), bins=50)
