# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 11:53:18 2015

@author: nicoguaro_2
"""
from __future__ import division
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import sys
sys.path.append(".\..")
from ellipse_packing import multi_subdivide


#%% Generation of the new polygon
nsides = 6
theta = np.linspace(0, 2*np.pi, nsides, endpoint=False) + \
        0.5*np.random.rand(nsides)
x = np.cos(theta)
y = np.sin(theta)
weights = [1, 6, 1]
xnew, ynew = multi_subdivide(x, y, 10, weights)


#%% Plotting
fig = plt.figure()  
ax = fig.add_subplot(111, aspect='equal')
poly = Polygon(np.column_stack([x,y]), facecolor="gray")
poly2 = Polygon(np.column_stack([xnew, ynew]), facecolor="w")
ax.add_artist(poly)
plt.plot(x, y, 'ok')
ax.add_artist(poly2)
plt.xlim(1.2*np.min(x), 1.2*np.max(x))
plt.ylim(1.2*np.min(y), 1.2*np.max(y))
plt.show()