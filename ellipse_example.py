# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:30:25 2015

@author: nguarin
"""
from __future__ import division
import numpy as np
from ellipse_packing import steiner_inellipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay, delaunay_plot_2d


#%% Delaunay and Voronoi
np.random.seed(3)
pts = np.random.rand(200, 2)
vor = Voronoi(pts)
voronoi_plot_2d(vor)
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
tri = Delaunay(pts)
delaunay_plot_2d(tri, ax=ax)
for triang in tri.simplices:
    A = triang[0]
    B = triang[1]
    C = triang[2]
    vertices = np.array([pts[A], pts[B], pts[C]])
    centroid, semi_minor, semi_major, ang = steiner_inellipse(vertices)
    ellipse = Ellipse(centroid, 1.8*semi_major, 1.8*semi_minor, angle=ang,
                      facecolor="red")
    ax.add_artist(ellipse)

plt.show()