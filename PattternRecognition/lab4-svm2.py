import numpy as np
import random
import matplotlib.pyplot as plt

x = [random.randint(1, 50) for n in range(100)]
y = [random.randint(1, 50) for n in range(100)]
plt.scatter(x, y);
plt.show

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=1)
classifier.fit(x, y)
