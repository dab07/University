# import matplotlib.pyplot as plt
# import random
# from scipy.optimize import curve_fit
#
# x = [random.randint(1, 100) for n in range(100)]
# y = [random.randint(1, 100) for n in range(100)]
#
# param, param_cov = curve_fit(test, x, y)
#
# def test(x, a, b):
#     return a * np.sin(b * x)
#
# plt.scatter(x, y)
# plt.show()

import numpy as np
from scipy import linalg
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import style




