# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:30:25 2015

@author: nguarin
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay, delaunay_plot_2d
sys.path.append(".\..")
from ellipse_packing.ellipse_packing import steiner_inellipse


#%% Delaunay and Voronoi
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
np.random.seed(3)
pts = np.random.rand(100, 2)
vor = Voronoi(pts)
voronoi_plot_2d(vor, ax=ax)
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