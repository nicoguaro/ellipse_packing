# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:39:34 2015

@author: nguarin
"""
from __future__ import division
import numpy as np
import sys
sys.path.append(".\..")
from ellipse_packing import delaunay_ellipses
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


#%% Delaunay and Voronoi
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
np.random.seed(3)
nx = 50
ny = 10
xmin = 0
xmax = 5
ymin = 0
ymax = 1
x, y = np.mgrid[xmin:xmax:nx*1j,
                ymin:ymax:ny*1j]
x[:, 1::2] = x[:, 1::2] + 1/nx
x.shape = (nx*ny, 1)
y.shape = (nx*ny, 1)
pts = np.hstack([x, y]) + 0.02*np.random.normal(size=(nx*ny, 2))
scal = 0.8
del_ellipses = delaunay_ellipses(pts)
for ellipse in del_ellipses:
    centroid, semi_minor, semi_major, ang = ellipse
    ellipse = Ellipse(centroid, 2*scal*semi_major, 2*scal*semi_minor,
                      angle=ang, facecolor="blue", alpha=0.4)
    ax.add_artist(ellipse)

plt.xlim(np.min(pts[:,0]), np.max(pts[:,0]))
plt.ylim(np.min(pts[:,1]), np.max(pts[:,1]))
plt.show()