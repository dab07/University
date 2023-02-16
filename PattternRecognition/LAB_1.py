import numpy as np

from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

d = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                   'y': [3, 14, 23, 25, 23, 15, 9, 5, 9, 13, 17, 24, 32, 36, 40, 38, 35, 32, 27, 22]})

plt.scatter(d.x, d.y)

model1 = np.poly1d(np.polyfit(d.x, d.y, 1))
model2 = np.poly1d(np.polyfit(d.x, d.y, 2))
model3 = np.poly1d(np.polyfit(d.x, d.y, 3))
model4 = np.poly1d(np.polyfit(d.x, d.y, 4))
model5 = np.poly1d(np.polyfit(d.x, d.y, 5))
model6 = np.poly1d(np.polyfit(d.x, d.y, 6))


poly = np.linspace(1, 20, 50)
plt.scatter(d.x, d.y)

plt.plot(poly, model1(poly), color='red')
plt.plot(poly, model2(poly), color='blue')
plt.plot(poly, model3(poly), color='green')
plt.plot(poly, model4(poly), color='violet')
plt.plot(poly, model5(poly), color='yellow')
plt.plot(poly, model6(poly), color='gray')

plt.show()

def adjR(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = 1- (((1-(ssreg/sstot))*(len(y)-1))/(len(y)-degree-1))
    print(results)
    return results

adjR(d.x, d.y, 1)
adjR(d.x, d.y, 2)
adjR(d.x, d.y, 3)
adjR(d.x, d.y, 4)
adjR(d.x, d.y, 5)
adjR(d.x, d.y, 6)





