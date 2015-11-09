# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:39:34 2015

@author: nguarin
"""
from __future__ import division
import os, sys
sys.path.append(os.path.dirname(__file__ ) + "\..")
from ellipse_packing import multi_subdivide
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import  Polygon
from scipy.spatial import Voronoi

def voronoi_polygons(pts):
    """Polygons from the Voronoi tesselation of a pointset
    
    
    """
    vor = Voronoi(pts)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    polys = []
    weights = [1, 6, 1]
    for poly in vor.regions:
        vertices = np.array(vor.vertices[poly])
        if -1 in poly or len(poly)==0:
            pass
        elif (vertices[:,0]<xmin).any() or (vertices[:,1]<ymin).any() or \
             (vertices[:,0]>xmax).any() or (vertices[:,1]>ymax).any():
            pass
        else:
            vertices.shape = (len(poly), 2)
            x, y = multi_subdivide(vertices[:, 0], vertices[:, 1], 3, weights)
            polys.append(np.column_stack([x,y]))

    return polys



#%% Delaunay and Voronoi
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
np.random.seed(3)
nx = 50
ny = 10
xmin = 0
xmax = 5
ymin = 0
ymax = np.sqrt(2)/2
x, y = np.mgrid[xmin:xmax:nx*1j,
                ymin:ymax:ny*1j]
x[:, 1::2] = x[:, 1::2] + (xmax - xmin)/(2*nx)
x.shape = (nx*ny, 1)
y.shape = (nx*ny, 1)
pts = np.hstack([x, y]) + 0.005*np.random.normal(size=(nx*ny, 2))
scal = 0.8
vor_polys = voronoi_polygons(pts)
for poly in vor_polys:
    ax.add_artist(Polygon(poly, facecolor="green", alpha=0.4))

plt.xlim(np.min(pts[:,0]), np.max(pts[:,0]))
plt.ylim(np.min(pts[:,1]), np.max(pts[:,1]))
plt.show()