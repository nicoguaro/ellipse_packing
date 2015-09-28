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
from scipy.spatial import Voronoi, Delaunay
#from scipy.spatial import voronoi_plot_2d, delaunay_plot_2d

def steiner_inellipse(vertices):
    r"""Compute the Steiner inellipse of a triangle
    
    The Steiner inellipse is the unique ellipse inscribed in the
    triangle and tangent to the sides at their midpoints. The lengths
    of the semi-major and semi-minor axes for a triangle with sides
    :math:`A`,  :math:`B`,  :math:`C` are [1]_
    
    .. math::
        \frac{1}{6}\sqrt{A^2 + B^2 + C^2 \pm 2Z}\ ,
        
    where :math:`Z = \sqrt{A^4 + B^4 + C^4 - (AB)^2 - (BC)^2 - (CA)^2}`.
    
    The angle is found using [2]_
    
    .. math::
        g \pm \sqrt{g^2 - \frac{1}{3}(z_1 z_2 + z_2 z_3 + z_1 z_3)}\ ,
        
    where :math:`g = \dfrac{1}{3}(z_1 + z_2 + z_3)` is the centroid,
    and :math:`z_i` are the vertices represented as complex numbers.

    
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


def poly_ellipse(vertices):
    """Return an ellipse given a polygon using SVD

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

    Examples
    --------

    >>> pts = np.array([[2, 0],
    ...                [0, 1],
    ...                [-2, 0],
    ...                [0, -1]])
    
    >>> centroid, semi_minor, semi_major, ang = poly_ellipse(pts)

    >>> print(np.isclose(centroid, [0, 0]))
    [ True  True]

    >>> print(np.isclose(semi_minor, np.sqrt(2/3)))
    True

    >>> print(np.isclose(semi_major, np.sqrt(8/3)))
    True

    >>> print(np.isclose(ang, 0))
    True


    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    cov = np.cov(vertices.T)
    vals, vecs = eigsorted(cov)
    ang = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    semi_major, semi_minor = sqrt(vals)        
    centroid = np.mean(vertices, axis=0)
    
    return centroid, semi_minor, semi_major, ang


def voronoi_ellipses(pts, min_aspect_ratio=0.2):
    """Ellipses from the Voronoi tesselation of a pointset
    
    
    """
    vor = Voronoi(pts)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    ellipses = []
    for poly in vor.regions:
        vertices = np.array(vor.vertices[poly])
        if -1 in poly or len(poly)==0:
            pass
        elif (vertices[:,0]<xmin).any() or (vertices[:,1]<ymin).any() or \
             (vertices[:,0]>xmax).any() or (vertices[:,1]>ymax).any():
            pass
        else:
            vertices.shape = (len(poly), 2)
            centroid, semi_minor, semi_major, ang = poly_ellipse(vertices)
            if semi_minor/semi_major > min_aspect_ratio:
                ellipses.append([centroid, semi_minor, semi_major, ang])

    return ellipses
    
    
def delaunay_ellipses(pts, min_aspect_ratio=0.2):
    """
    """
    tri = Delaunay(pts)
    ellipses = []
    for triang in tri.simplices:
        A = triang[0]
        B = triang[1]
        C = triang[2]
        vertices = np.array([pts[A], pts[B], pts[C]])
        centroid, semi_minor, semi_major, ang = steiner_inellipse(vertices)
        if semi_minor/semi_major > min_aspect_ratio:
            ellipses.append([centroid, semi_minor, semi_major, ang])
        
    return ellipses


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    # Triangle inellipse
    pts = np.array([[1, 7],
                    [7, 5],
                    [3, 1]])              
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    centroid, semi_minor, semi_major, ang = steiner_inellipse(pts)
    ellipse = Ellipse(centroid, 2*semi_major, 2*semi_minor, angle=ang,
                      alpha=0.2)
    poly = Polygon(pts, fill=False)
    ax.add_artist(ellipse)
    ax.add_artist(poly)
    plt.xlim(np.min(pts[:,0]), np.max(pts[:,0]))
    plt.ylim(np.min(pts[:,1]), np.max(pts[:,1]))

    # Rhombic quadrilateral
    theta = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    pts = np.empty((4, 2))
    pts[:, 0] = 2*np.cos(theta)
    pts[:, 1] = np.sin(theta)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    centroid, semi_minor, semi_major, ang = poly_ellipse(pts)
    ellipse = Ellipse(centroid, 2*semi_major, 2*semi_minor, angle=ang,
                      alpha=0.2)
    poly = Polygon(pts, fill=False)
    ax.add_artist(ellipse)
    ax.add_artist(poly)
    plt.xlim(np.min(pts[:,0]), np.max(pts[:,0]))
    plt.ylim(np.min(pts[:,1]), np.max(pts[:,1]))
    
    # Random polygon    
    nsides = np.random.random_integers(4, 8)
    theta = np.linspace(0, 2*np.pi, nsides, endpoint=False) \
        + np.pi/20*np.random.normal(size=nsides)
    pts = np.empty((nsides, 2))
    pts[:, 0] = 2*np.cos(theta) + 0.2*np.random.normal(size=nsides)
    pts[:, 1] = np.sin(theta) + 0.1*np.random.normal(size=nsides)
    x = pts[:,0]
    y = pts[:,1]
    
    fig = plt.figure()  
    ax = fig.add_subplot(111, aspect='equal')
    
    centroid, semi_minor, semi_major, ang = poly_ellipse(pts)
    ellipse = Ellipse(centroid, 2*semi_major, 2*semi_minor, angle=ang,
              alpha=0.2, lw=0)
    poly = Polygon(pts, fill=False)
    ax.add_artist(ellipse)
    ax.add_artist(poly)
    plt.xlim(np.min(pts[:,0]), np.max(pts[:,0]))
    plt.ylim(np.min(pts[:,1]), np.max(pts[:,1]))
    
    plt.show()