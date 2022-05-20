import pinocchio as pin
import hppfcl
import numpy as np
import sys
import os
from os.path import dirname, join, abspath

from pinocchio.visualize import MeshcatVisualizer

pinocchio_model_dir = join("/home/cbouchah/devel", "pinocchio/models")
model_path = join(pinocchio_model_dir, "example-robot-data/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "talos_full_v2.urdf"
urdf_model_path = join(join(model_path, "talos_data/robots"), urdf_filename)

model, col_model, viz_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())

viz = MeshcatVisualizer(model, col_model, viz_model)

q0 = pin.neutral(model)
q0[40] = np.pi/4
viz.initViewer()
viz.loadViewerModel()
viz.display(q0)

model2 = pin.Model()

geom_model = pin.GeometryModel()
geometries = [
    # hppfcl.Capsule(0.1, 0.8),
    hppfcl.Sphere(0.1),
    # hppfcl.Box(1, 1, 1),
    # hppfcl.Cylinder(0.1, 1.0),
    # hppfcl.Cone(0.5, 1.0),
]

for i, geom in enumerate(geometries):
    placement = pin.SE3(np.eye(3), np.array([0.14, -0.27, 0.75]))
    geom_obj = pin.GeometryObject("obj{}".format(i), 0, 0, geom, placement)
    color = np.random.uniform(0, 1, 4)
    color[3] = 1
    geom_obj.meshColor = color
    geom_model.addGeometryObject(geom_obj)

viz2 = MeshcatVisualizer(model2, geom_model, geom_model)
viz2.initViewer()
viz2.loadViewerModel()
viz2.display(np.zeros(0))




