import speedup_griddata as sg
import numpy as np

x = np.linspace(0, 100, 10)
y = np.linspace(0, 100, 10)
x, y = np.meshgrid(x, y)
value = np.random.rand(x.shape[0], y.shape[0])
xnew = np.linspace(10, 90, 20)
ynew = np.linspace(10, 90, 20)
xnew, ynew = np.meshgrid(xnew, ynew)

vtx, wts = sg.interp_weights(x, y, xnew, ynew)

value_new = sg.interpolate(value, vtx, wts, shape = xnew.shape)

import matplotlib.pyplot as plt
plt.subplot(121)
plt.contourf(value, vmin = 0, vmax = 1)
plt.subplot(122)
plt.contourf(value_new, vmin = 0, vmax = 1)

