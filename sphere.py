import pinocchio as pin
import hppfcl
import numpy as np
import sys
import os
from os.path import dirname, join, abspath
import time
from pinocchio.visualize import MeshcatVisualizer

model2 = pin.Model()

geom_model = pin.GeometryModel()
geometries = [hppfcl.Sphere(0.1)]

    placement = pin.SE3(np.eye(3), np.array([0.33, -0.25, 1.11]))
    geom_obj = pin.GeometryObject("obj{}".format(i), 0, 0, geometries, placement)
    color = np.random.uniform(0, 1, 4)
    color[3] = 1
    geom_obj.meshColor = color
    geom_model.addGeometryObject(geom_obj)

viz2 = MeshcatVisualizer(model2, geom_model, geom_model)
viz2.initViewer()
viz2.loadViewerModel()
viz2.display(np.zeros(0))
