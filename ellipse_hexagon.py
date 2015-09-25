# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 01:07:36 2015

@author: nicoguaro_2
"""
from __future__ import division
import numpy as np
from numpy import sin, cos, sqrt, pi, empty, linspace
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


fig = plt.figure()
nsides = 7
ax = fig.add_subplot(111, aspect='equal')
np.random.seed(2)
theta = linspace(0, 2*pi, nsides, endpoint=False) \
        + pi/20*np.random.normal(size=nsides)
print theta
points = empty((nsides, 2))
points[:, 0] = 2*cos(theta) + 0.2*np.random.normal(size=nsides)
points[:, 1] = sin(theta) + 0.1*np.random.normal(size=nsides)
x = points[:,0]
y = points[:,1]
plt.plot(x, y, 'o')

cov = np.cov(x, y)
vals, vecs = eigsorted(cov)
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
w, h = 2 * sqrt(vals)        
center = np.mean(points, 0)
ell = Ellipse(center, width=w, height=h, angle=theta, alpha=0.2,
                      lw=0)
ax.add_artist(ell)