# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:39:34 2015

@author: nguarin
"""
from __future__ import absolute_import, division, print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import  Polygon
sys.path.append("./..")
from ellipse_packing.ellipse_packing import voronoi_poly


#%% Delaunay and Voronoi
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
np.random.seed(3)
nx = 50
ny = 10
xmin = -2.5
xmax = 2.5
ymin = -np.sqrt(2)/4
ymax = np.sqrt(2)/4
x, y = 1.1*np.mgrid[xmin:xmax:nx*1j,
                ymin:ymax:ny*1j]
x[:, 1::2] = x[:, 1::2] + (xmax - xmin)/(2*nx)
x.shape = (nx*ny, 1)
y.shape = (nx*ny, 1)
pts = np.hstack([x, y]) + 0.01*np.random.normal(size=(nx*ny, 2))
vor_polys = voronoi_poly(pts, scaling=0.95)
for poly in vor_polys:
    ax.add_artist(Polygon(poly, facecolor="green", alpha=0.4))

plt.hlines(ymin, xmin, xmax)
plt.hlines(ymax, xmin, xmax)
plt.vlines(xmin, ymin, ymax)
plt.vlines(xmax, ymin, ymax)

plt.xlim(np.min(pts[:,0]), np.max(pts[:,0]))
plt.ylim(np.min(pts[:,1]), np.max(pts[:,1]))
plt.show()