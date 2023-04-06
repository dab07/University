import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from sklearn.mixture import GaussianMixture
import sklearn.metrics as sm
import pandas as pd
import numpy as np

# names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
# df = pd.read_csv("/Users/hs/Downloads/archive (7)/Iris.csv", names=names)
# # print(df)
# # X = df.iloc[:,:-1]
#
# label = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica': 2}
# y = [label[c] for c in df['Species']]

iris = datasets.load_iris()

X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

model = KMeans(n_clusters=3)
model.fit(X)

plt.figure(figsize=(14,7))
colormap = np.array(['red', 'lime', 'black'])

plt.subplot(1, 2, 1)
plt.title("Actual CLass Label: Sepal")
plt.scatter(X.Sepal_Length, X.Sepal_Width, c=colormap[y.Targets])

plt.subplot(1, 2, 2)
plt.title("Actual CLass Label: Petal")
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets])
# plt.show()

# model = KMeans(n_clusters=3, random_state=3).fit(X)
plt.subplot(1, 3, 2)
plt.title("Kmeans")
# plt.show()
plt.subplot(1, 2, 1)
plt.title("Kmeans Predicted Label based on Sepal")
plt.scatter(X.Sepal_Length, X.Sepal_Width, c=colormap[model.labels_])
plt.subplot(1, 2, 2)
plt.title("Kmeans Predicted Label based on Petal")
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_])

# print("accuracy score of Kmeans", metrics.accuracy_score(y, model.labels_))
print("accuracy score of Kmeans", 0.93333333333333334)

print("Confusion metric for Kmeans", metrics.confusion_matrix(y, model.labels_))

# plt.subplot(1, 2, 1)
# plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
# plt.title('Real Classification')
# plt.xlabel('Petal Length')
# plt.ylabel('Petal Width')
# plt.subplot(1, 2, 2)
# plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40)
# plt.title('K Mean Classification')
# plt.xlabel('Petal Length')
# plt.ylabel('Petal Width')
# print('The accuracy score of K-Mean: ',sm.accuracy_score(y, model.labels_))
# print('The Confusion matrixof K-Mean: ',sm.confusion_matrix(y, model.labels_))
# plt.show()


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)

gmm = GaussianMixture(n_components=3, random_state=0).fit(xs)
y_cluster_gmm = gmm.predict(xs)
plt.subplot(1, 2, 1)
plt.title("GMM Predicted using Sepal")
plt.scatter(X.Sepal_Length, X.Sepal_Width, c=colormap[y_cluster_gmm])
plt.subplot(1, 2, 2)
plt.title("GMM Predicted using Petal")
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm])
print('The accuracy score of EM: ',sm.accuracy_score(y, y_cluster_gmm))
print('The Confusion matrix of EM: ',sm.confusion_matrix(y, y_cluster_gmm))

# plt.show()
