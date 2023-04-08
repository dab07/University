import pandas as pd
import numpy as np
from sklearn.utils import column_or_1d

df = pd.read_csv("/Users/hs/Downloads/diabetes2.csv")
# print(df.columns)
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]
y = df.iloc[:, 8:9]
y = column_or_1d(y, warn=True)
X_train, X_test, y_train,y_test = train_test_split(X, y, train_size=0.7, random_state=0)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=1)
classifier.fit(X_train, y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)
accuracy = float(cm.diagonal().sum())/len(y_test)
print("\nAccuracy: ", accuracy)
