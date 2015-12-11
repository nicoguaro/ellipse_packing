# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:17:56 2015

@author: Nicolas Guarin-Zapata
"""
from __future__ import division
import numpy as np
from numpy import sqrt, angle
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from scipy.spatial import Voronoi, Delaunay

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


def split_average(x, y, w=(1, 1, 1)):
    """Compute an iteration of the subdivision algorithm [1]_
    
    Parameters
    ----------
    x : ndarray (n)
        Unidimensional array with the x coordinates for the polygon.
    y : ndarray (n)
        Unidimensional array with the y coordinates for the polygon.
    w : tuple (3, optional)
        Weights for the points. They should have the same sign for
        *normal* uses. And the sum should not be zero.

    Returns
    -------
    xnew : ndarray (2*n)
        Unidimensional array with the x coordinates for the subdivided
        polygon.
    ynew : ndarray (2*n)
        Unidimensional array with the y coordinates for the subdivided
        polygon.

    References
    ----------
     .. [1] Catmull, Edwin. A subdivision algorithm for computer
         display of curved surfaces. No. UTEC-CSC-74-133. UTAH
         UNIV SALT LAKE CITY SCHOOL OF COMPUTING, 1974.

    Examples
    --------
    
    >>> x = [1, 0, -1, 0]
    >>> y = [0, 1, 0, -1]
    >>> xnew, ynew = split_average(x, y)
    >>> print(np.round(xnew, 4))
    [ 0.6667  0.5     0.     -0.5    -0.6667 -0.5     0.      0.5   ]
    >>> print(np.round(ynew, 4))
    [ 0.      0.5     0.6667  0.5     0.     -0.5    -0.6667 -0.5   ]

    """
    n = len(x)
    xnew = np.zeros((2*n))
    ynew = np.zeros((2*n))
    xnew[::2] = x
    ynew[::2] = y
    xnew[1::2] = [0.5*(x[k] + x[(k+1)%n]) for k in range(n)]
    ynew[1::2] = [0.5*(y[k] + y[(k+1)%n]) for k in range(n)]
    xnew[::2] = [(w[0]*xnew[2*k-1] + w[1]*xnew[2*k] + w[2]*xnew[(2*k+1)%(2*n)])
                    /(w[0] + w[1] + w[2]) for k in range(n)]
    ynew[::2] = [(w[0]*ynew[2*k-1] + w[1]*ynew[2*k] + w[2]*ynew[(2*k+1)%(2*n)])
                    /(w[0] + w[1] + w[2]) for k in range(n)]
    return xnew, ynew


def multi_subdivide(x, y, times, weights=(1,1,1)):
    """Apply multiple iteration of the subdivision algorithm
    
    Parameters
    ----------
    x : ndarray (n)
        Unidimensional array with the x coordinates for the polygon.
    y : ndarray (n)
        Unidimensional array with the y coordinates for the polygon.
    times : int
        Number of iterations for the subdivision algorithm.
    weights : tuple (3, optional)
        Weights for the points. They should have the same sign for
        *normal* uses. And the sum should not be zero.

    Returns
    -------
    x : ndarray (2*n)
        Unidimensional array with the x coordinates for the subdivided
        polygon.
    y : ndarray (2*n)
        Unidimensional array with the y coordinates for the subdivided
        polygon.
    """
    for k in range(times):
        x, y = split_average(x, y, w=weights)
        
    return x, y


def voronoi_poly(pts, scaling=0.9):
    """Polygons from the Voronoi tesselation of a pointset

    Parameters
    ----------
    pts : array like (npts, 2)
        Set of points to compute the Voronoi tesselation.
    scaling : float (>=0 and <=1)
        Scale factor to the polygons forming the Voronoi tesselation.

    Returns
    -------
    polys : list
        List of vertices of the polygons forming the Voronoi
        tesselation.

    """
    vor = Voronoi(pts)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    polys = []
    for poly in vor.regions:
        vertices = np.array(vor.vertices[poly])
        if -1 in poly or len(poly)==0:
            pass
        elif (vertices[:,0]<xmin).any() or (vertices[:,1]<ymin).any() or \
             (vertices[:,0]>xmax).any() or (vertices[:,1]>ymax).any():
            pass
        else:
            vertices.shape = (len(poly), 2)
            mean = np.mean(vertices, axis=0)
            vertices = scaling * (vertices - mean) + mean
            polys.append(vertices)

    return polys


def voronoi_smooth_poly(pts, niter=3, weigths=[1, 6, 1], scaling=1.0):
    """Smoothed polygons from the Voronoi tesselation of a pointset
    
    The smoothing is done with subdivision algorithm.

    Parameters
    ----------
    pts : array like (npts, 2)
        Set of points to compute the Voronoi tesselation.
    niter : int
        Number of subdivisions to use.
    weights : list
        Weights for the the corners of the smoothing process.
    scaling : float (>=0 and <=1)
        Scale factor to the polygons forming the Voronoi tesselation.

    Returns
    -------
    polys : list
        List of vertices of the smoothed polygons forming the
        Voronoi tesselation.

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
            mean = np.mean(vertices, axis=0)
            vertices = scaling * (vertices - mean) + mean
            x, y = multi_subdivide(vertices[:, 0], vertices[:, 1], niter,
                                   weights)
            polys.append(np.column_stack([x,y]))

    return polys


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    from matplotlib import rcParams

    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 16

    # Examples
    # --------

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
    plt.title("Steiner inellipse")

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
    plt.title("Ellipse in a rhombic quadrilateral")


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
    ellipse = Ellipse(centroid, 1.8*semi_major, 1.8*semi_minor, angle=ang,
              alpha=0.2, lw=0)
    poly = Polygon(pts, fill=False)
    ax.add_artist(ellipse)
    ax.add_artist(poly)
    plt.xlim(np.min(pts[:,0]), np.max(pts[:,0]))
    plt.ylim(np.min(pts[:,1]), np.max(pts[:,1]))
    plt.title("Ellipse in a random polygon")


    # Subdivision algorithm
    nsides = 6
    theta = np.linspace(0, 2*np.pi, nsides, endpoint=False) + \
            0.5*np.random.rand(nsides)
    x = np.cos(theta)
    y = np.sin(theta)
    weights = [1, 20, 1]
    xnew, ynew = multi_subdivide(x, y, 4, weights)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    poly = Polygon(np.column_stack([x,y]), fill=False)
    poly2 = Polygon(np.column_stack([xnew, ynew]), alpha=0.2, lw=0)
    ax.add_artist(poly)
    ax.add_artist(poly2)
    plt.xlim(1.2*np.min(x), 1.2*np.max(x))
    plt.ylim(1.2*np.min(y), 1.2*np.max(y))
    plt.title("Subdivision algorithm in a polygon")
    
    plt.show()