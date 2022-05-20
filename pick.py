import pinocchio as pin
import hppfcl
import numpy as np
import sys
import os
from os.path import dirname, join, abspath

from pinocchio.visualize import MeshcatVisualizer

model2 = pin.Model()

geom_model = pin.GeometryModel()
geometries = [
    hppfcl.Sphere(0.5),
]

for i, geom in enumerate(geometries):
    placement = pin.SE3(np.eye(3), np.array([.5, .1, .2]))
    geom_obj = pin.GeometryObject("obj{}".format(i), 0, 0, geom, placement)
    color = np.random.uniform(0, 1, 4)
    color[3] = 1
    geom_obj.meshColor = color
    geom_model.addGeometryObject(geom_obj)

viz2 = MeshcatVisualizer(model2, geom_model, geom_model)
viz2.initViewer()
viz2.loadViewerModel()
viz2.display(np.zeros(0))



model = pin.buildSampleModelHumanoid()
visual_model = pin.buildSampleGeometryModelHumanoid(model)
collision_model = visual_model.copy()

viz = MeshcatVisualizer(model, collision_model, visual_model)


q= pin.randomConfiguration(model)
q[0:3] = 0
data=pin.createDatas(model)
visual_data=pin.createDatas(visual_model)

# pin.forwardKinematics(model, data, q)
# pin.updateGeometryPlacements(model, data, visual_model, visual_data)


viz.initViewer()
viz.loadViewerModel()
viz.display(q)

print()

