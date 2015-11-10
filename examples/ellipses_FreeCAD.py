from __future__ import division
import numpy as np
import FreeCAD as FC
import Part
import Draft
import os


doc = FC.newDocument("ellipses")
folder = os.path.dirname(__file__) + ".\.."
fname = folder + "/vor_ellipses.txt"
data = np.loadtxt(fname)
shapes = []
area = 0
for ellipse in data:
    cx, cy, b, a, ang = ellipse
    ang = ang*np.pi/180
    place = FC.Placement()
    place.Rotation = (0, 0, np.sin(ang/2), np.cos(ang/2))
    place.Base = FC.Vector(100*cx, 100*cy, 0)
    ellipse = Draft.makeEllipse(100*a, 100*b, placement=place)
    shapes.append(ellipse)
    area = area + np.pi*a*b*100*100

print area, " ", area/500/70
part = Part.makeCompound(shapes)