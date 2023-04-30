from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

breast_cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2, random_state=0)

svm = SVC(kernel='rbf', random_state=1)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred, target_names=breast_cancer.target_names))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = float(cm.diagonal().sum())/len(y_test)
print("\nAccuracy: ", accuracy)
