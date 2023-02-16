import classifier as classifier
import pandas as pd

dataset = pd.read_csv("/Users/hs/Downloads/diabetes2.csv")
# print(dataset.head())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 8:9]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn import model_selection
from sklearn.model_selection import cross_val_score

kfold = model_selection.KFold(n_splits=9)

modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))