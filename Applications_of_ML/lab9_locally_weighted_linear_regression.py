import numpy as np
import matplotlib.pyplot as plt

np.random.seed(8)
X = np.random.randn(1000,1)
y = 2*(X**3) + 10 + 4.6*np.random.randn(1000,1)


def wm(point, X, tau):
    m = X.shape[0]
    w = np.mat(np.eye(m))
    for i in range(m):
        xi = X[i]
        d = (-2 * tau * tau)
        w[i, i] = np.exp(np.dot((xi - point), (xi - point).T) / d)
    return w


def predict(X, y, point, tau):
    m = X.shape[0]
    X_ = np.append(X, np.ones(m).reshape(m, 1), axis=1)
    point_ = np.array([point, 1])
    w = wm(point_, X_, tau)
    theta = np.linalg.pinv(X_.T * (w * X_)) * (X_.T * (w * y))
    pred = np.dot(point_, theta)
    return theta, pred


def plot_predictions(X, y, tau, nval):
    X_test = np.linspace(-3, 3, nval)

    preds = []

    for point in X_test:
        theta, pred = predict(X, y, point, tau)
        preds.append(pred)

    X_test = np.array(X_test).reshape(nval, 1)
    preds = np.array(preds).reshape(nval, 1)

    plt.plot(X, y, 'b.')
    plt.plot(X_test, preds, 'r.')  # Predictions in red color.
    plt.show()


plot_predictions(X, y, 0.08, 100)