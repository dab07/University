import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

df = pd.read_csv("/Users/hs/UNI_Material/Datasets/Iris.csv");
X = df.iloc[:,:-1]
Y = df.iloc[:,5:6]

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0)
DTC = DecisionTreeClassifier(criterion="gini", min_samples_split=2)
DTC.fit(train_x, train_y)
print(plot_tree(DTC))

tree_graph = tree.export_text(DTC)
graph = graphviz.Source(tree_graph)
print(graph)

y_pred = DTC.predict(test_x)
print("Accuracy of test data", accuracy_score(test_y, y_pred))

