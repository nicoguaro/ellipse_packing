# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:39:34 2015

@author: nguarin
"""
from __future__ import division
import numpy as np
from ellipse_packing import steiner_inellipse, poly_ellipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay, delaunay_plot_2d


#%% Delaunay and Voronoi
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
np.random.seed(3)
x, y = np.mgrid[0:1:10j, 0:1:10j]
x[:, 1::2] = x[:, 1::2] + 0.05
x.shape = (100, 1)
y.shape = (100, 1)
pts = np.hstack([x, y]) + 0.01*np.random.normal(size=(100, 2))
vor = Voronoi(pts)
voronoi_plot_2d(vor, ax=ax)
tri = Delaunay(pts)
#delaunay_plot_2d(tri, ax=ax)
scal = 0.9
for triang in tri.simplices:
    A = triang[0]
    B = triang[1]
    C = triang[2]
    vertices = np.array([pts[A], pts[B], pts[C]])
    centroid, semi_minor, semi_major, ang = steiner_inellipse(vertices)
    ellipse = Ellipse(centroid, 2*scal*semi_major, 2*scal*semi_minor,
                      angle=ang, facecolor="red", alpha=0.4)
    ax.add_artist(ellipse)
    
for poly in vor.regions:
    if -1 in poly or len(poly)==0:
        pass
    else:
        vertices = np.array(vor.vertices[poly])
        vertices.shape = (len(poly), 2)
        centroid, semi_minor, semi_major, ang = poly_ellipse(vertices)
        ellipse = Ellipse(centroid, scal*semi_major, scal*semi_minor,
                          angle=ang, facecolor="green", alpha=0.4)
        ax.add_artist(ellipse)

plt.show()