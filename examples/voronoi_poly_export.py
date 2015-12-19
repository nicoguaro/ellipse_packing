# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:39:34 2015

@author: nguarin
"""
from __future__ import division
import os, sys
sys.path.append(os.path.dirname(__file__ ) + "\..")
from ellipse_packing import voronoi_poly
import numpy as np
#import FreeCAD as FC
#import Part


#%% Delaunay and Voronoi
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
fid = open("voronoi_poly.txt", "w")
for poly in vor_polys:
    poly = poly.flatten()
    text = ("%s " * (len(poly) - 1) + "%s\n")  % tuple(poly)
    fid.write(text)
fid.close()