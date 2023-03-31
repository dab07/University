import scipy.optimize as optimize
import numpy as np
import collections
import matplotlib.pyplot as plt
from numpy import random


def sampler(points, mu, sigma):
    return np.linspace(0, 2 * np.pi, points), \
           np.sin(np.linspace(0, 2 * np.pi, points)) \
           + random.normal(mu, sigma, size=(points))


def fitting_curve(param, x):
    A, B, C = param
    y = A * np.sin(B * x) + C
    return y


def error(param, x, y):
    return y - fitting_curve(param, x)


def fitting(mu, sigma):
    x, y = sampler(100, mu, sigma)
    p_guess = (1, 1, 1)
    fit_param = optimize.leastsq(error, p_guess, args=(x, y), full_output=True)[0]
    return fit_param


mu = 0
sigma = 0.1
def curvefit_routine(mu, sigma):
	#for g in [x for x in np.arange(0.01,1,0.1)]:
	fit_param = fitting(mu,sigma)
	x , y = sampler(100,mu,sigma)
	xfit = np.linspace(0, 2*np.pi, 1500)
	yfit=fitting_curve(fit_param,xfit)
	print "A = {0}, B = {1}, C = {2}".format(fit_param[0],fit_param[1],fit_param[2])
	A, B, C = fit_param[0],fit_param[1],fit_param[2]
	plt.plot(x, y, '.')
	plt.plot(xfit, yfit,label="Curve Fit")
	return A, B, C


def model_func(x, A, B, C, D):
    return np.exp(-(C * x + D)) * (A * x + B)


data = np.load("curve_fitting_data.dat")
x = [all[0] for all in data]
y = [all[1] for all in data]
params = optimize.curve_fit(model_func, x, y, p0=(1, 1, 1, 1))[0]

plt.plot(x, y, "o")
xfit = np.linspace(min(x), max(x), 1000)
plt.plot(xfit, model_func(xfit, params[0], params[1], params[2], params[3]))

main_list = []
iter = 100
shape = 20
for all in range(iter):
	lista, listb, listc = [], [], []
	for g in [x for x in np.linspace(0.01,1,shape)]:
		A, B, C = curvefit_routine(0,g)
		lista.append(A); listb.append(B); listc.append(C);
	main_list.append([lista,listb,listc])
aseries = np.zeros((shape))
bseries = np.zeros((shape))
cseries = np.zeros((shape))
for all in range(iter):
	aseries += main_list[all][0]
	bseries += main_list[all][1]
	cseries += main_list[all][2]

asdisp = np.zeros((shape))
bsdisp = np.zeros((shape))
csdisp = np.zeros((shape))
amean = aseries/iter
bmean = bseries/iter
cmean = cseries/iter
for all in range(shape):
	asdisp[all] = np.std([main_list[i][0][all] for i in range(iter)])
	bsdisp[all] = np.std([main_list[i][1][all] for i in range(iter)])
	csdisp[all] = np.std([main_list[i][2][all] for i in range(iter)])

plt.plot(np.linspace(0.01,1,shape), amean, "r",label = "A")
plt.plot(np.linspace(0.01,1,shape), bmean, "g", label = "B")
plt.plot(np.linspace(0.01,1,shape), cmean, "b", label = "C")

plt.errorbar(range(shape),cmean, yerr=asdisp,mfc='red', mec='red', label = "A")
plt.errorbar(range(shape),bmean, yerr=bsdisp,mfc='green', mec='green', label = "B")
plt.errorbar(range(shape),amean, yerr=csdisp,mfc='blue', mec='blue', label = "C")
plt.plot(np.linspace(0.01,1,shape), np.ones((shape)), "r--")
plt.plot(np.linspace(0.01,1,shape), np.ones((shape)), "g--")
plt.plot(np.linspace(0.01,1,shape), np.zeros((shape)), "b--")
plt.xlabel("mu")
plt.ylabel("Parameter Value")
plt.legend(loc="right")
plt.show()
