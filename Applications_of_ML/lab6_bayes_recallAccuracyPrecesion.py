import pandas as pd
import numpy as np

df = pd.read_csv("/Users/hs/Downloads/Naivetext.csv", names=['message','label'])

df['labelNum'] = df.label.map({'pos': 1, 'neg' : 0})
df = df.iloc[1:,:]
x = df.message
y = df.labelNum

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(x_train)
xtest_dtm = count_vect.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm,y_train)
predicted = clf.predict(xtest_dtm)

from sklearn import metrics
print("\n Accuracy of the classifer is", metrics.accuracy_score(y_test,predicted))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted)
print("Confusion Matrix: \n" , cm)

print("")
