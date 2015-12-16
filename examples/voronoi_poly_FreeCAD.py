from __future__ import division
import numpy as np
import FreeCAD as FC
import Part
#import Draft
import os


doc = FC.newDocument("voronoi")
folder = os.path.dirname(__file__)
fname = folder + "/voronoi_poly.txt"
textfile = open(fname, "r")
shapes = []
for line in textfile.read().split('\n')[:-1]:
    pts = np.array(line.split(' '), dtype=float)
    npts = len(pts)//2
    pts = [FC.Vector(pts[2*k], pts[2*k + 1], 0) for k in range(npts)]
    pts.append(pts[0])
    poly = Part.makePolygon(pts)
    shapes.append(poly)

part = Part.makeCompound(shapes)
Part.show(part)
