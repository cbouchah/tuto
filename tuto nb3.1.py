from __future__ import print_function
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from numpy.linalg import norm, pinv
from os.path import dirname, join, abspath
import pinocchio as pin
import time
from utils import display_sphere

des_placement = pin.SE3()
des_placement.translation = np.array([1, 1, 1])
des_placement.rotation= pin.utils.rpyToMatrix(np.pi, np.pi/4, np.pi)
viz2 = display_sphere(des_placement)

pinocchio_model_dir = join("/home/cbouchah/devel", "pinocchio/models")
model_path = join(pinocchio_model_dir, "example-robot-data/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "romeo_small.urdf"
urdf_model_path = join(join(model_path, "romeo_description/urdf"), urdf_filename)
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())
data, collision_data, visual_data = pin.createDatas(model, collision_model, visual_model)
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(viz2.viewer)
viz.loadViewerModel(color = [1, 2, 1.5, 2.1])
time.sleep(3)



JOINT_ID = 31
q= pin.neutral(model)
eps    = 1e-4
DT     = 1e-2
damp   = 1e-12

i=0
while True:
    # qtot=np.concatenate((np.array([0, 0, 0.840252, 0, 0, 0, 1]),q[7:38]),axis=None)
    pin.forwardKinematics(model,data,q)
    current=data.oMi[JOINT_ID]
    err= pin.log(current.actInv(des_placement)).vector
    if norm(err) < eps:
     success=True
     break
    if i >= 1000:
     success = False
     break
    J = pin.computeJointJacobian(model,data,q,JOINT_ID)
    v = pinv(J).dot(err)
    q = pin.integrate(model,q,v*DT)
    viz.display(q)
    if not i % 10:
        print('%d: error = %s' % (i, err.T))

    i += 1
print()


