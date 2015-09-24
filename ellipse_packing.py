# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:17:56 2015

@author: nguarin
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay, delaunay_plot_2d

pts = np.random.rand(100, 2)
vor = Voronoi(pts)
voronoi_plot_2d(vor)
tri = Delaunay(pts)
delaunay_plot_2d(tri)
plt.show()