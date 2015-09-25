# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:17:56 2015

@author: nguarin
"""
from __future__ import division
import numpy as np
from numpy import sqrt, angle
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay, delaunay_plot_2d

def steiner_inellipse(vertices):
    """Compute the Steiner inellipse of a triangle
    
    The Steiner inellipse is the unique ellipse inscribed in the
    triangle and tangent to the sides at their midpoints [1]_.
    The angle is found using the formula for the foci presented
    in [2]_.
    
    
    
    Parameters
    ----------
    vertices : ndarray (3, 2)
        Vertices of the triangle.

    Returns
    -------
    centroid : ndarray (2,)
        Centroid of the triangle.
    semi_minor : float (positive)
        Semi-minor axis of the ellipse.
    semi_major : float (positive)
        Semi-major axis of the ellipse.
    ang : float
        Angle (in degrees) with respect to the horizontal.

    References
    ----------
    .. [1] Wikipedia contributors. "Steiner inellipse." Wikipedia,
      The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 26 Jan.
      2015. Web. 24 Sep. 2015. 
    .. [2] Minda, David, and Steve Phelps. "Triangles, ellipses, and
      cubic polynomials." American Mathematical Monthly 115.8 (2008):
      679-689.

    Examples
    --------
    
    Let's try with a right isosceles triangle    
    
    >>> pts = np.array([[0, 0],
    ...                [0, 1],
    ...                [1, 0]])

    >>> centroid, semi_minor, semi_major, ang = steiner_inellipse(pts)

    >>> print(np.isclose(centroid, [1/3, 1/3]))
    [ True  True]

    >>> print(np.isclose(semi_minor, 0.2357))
    True

    >>> print(np.isclose(semi_major, 0.408248))
    True

    >>> print(np.isclose(ang, -45))
    True
    
    And Wikipedia's example

    >>> pts = np.array([[1, 7],
    ...                [7, 5],
    ...                [3, 1]])


    >>> centroid, semi_minor, semi_major, ang = steiner_inellipse(pts)
    >>> print(np.isclose(centroid, [11/3, 13/3]))
    [ True  True]

    >>> print(np.isclose(semi_minor, 1.632993))
    True

    >>> print(np.isclose(semi_major, 1.885618))
    True

    >>> print(np.isclose(ang, -45))
    True
        
    
    
    """
    # centroid
    centroid = np.mean(vertices, axis=0)
    
    # Semiaxes
    A = norm(vertices[0,:] - vertices[1,:])
    B = norm(vertices[1,:] - vertices[2,:])
    C = norm(vertices[2,:] - vertices[0,:])
    Z = sqrt(A**4 + B**4 + C**4 - (A*B)**2 - (B*C)**2 - (C*A)**2)
    semi_minor = 1./6.*sqrt(A**2 + B**2 + C**2 - 2*Z)
    semi_major = 1./6.*sqrt(A**2 + B**2 + C**2 + 2*Z)
    
    # Angle
    z1 = vertices[0,0] + 1j*vertices[0,1]
    z2 = vertices[1,0] + 1j*vertices[1,1]
    z3 = vertices[2,0] + 1j*vertices[2,1]
    g = 1/3*(z1 + z2 + z3)
    focus_1 = g + sqrt(g**2  - 1/3*(z1*z2 + z2*z3 + z3*z1))
    focus_2 = g - sqrt(g**2  - 1/3*(z1*z2 + z2*z3 + z3*z1))
    foci = focus_1 - focus_2
    ang = angle(foci, deg=True)
    return centroid, semi_minor, semi_major, ang





if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    pts = np.array([[1, 7],
                    [7, 5],
                    [3, 1]])
                 
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    centroid, semi_minor, semi_major, ang = steiner_inellipse(pts)
    ellipse = Ellipse(centroid, 2*semi_major, 2*semi_minor, angle=ang)
    poly = Polygon(pts, fill=False)
    ax.add_artist(ellipse)
    ax.add_artist(poly)
    plt.xlim(np.min(pts[:,0]), np.max(pts[:,0]))
    plt.ylim(np.min(pts[:,1]), np.max(pts[:,1]))
    plt.show()