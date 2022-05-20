from __future__ import print_function
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from numpy.linalg import norm, solve
from os.path import dirname, join, abspath
import pinocchio as pin
import time
pinocchio_model_dir = join("/home/cbouchah/devel", "pinocchio/models")
model_path = join(pinocchio_model_dir, "example-robot-data/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "romeo_small.urdf"
urdf_model_path = join(join(model_path, "romeo_description/urdf"), urdf_filename)
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())
data, collision_data, visual_data = pin.createDatas(model, collision_model, visual_model)
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer()
viz.loadViewerModel(color = [1, 2, 1.5, 2.1])


JOINT_ID = 6
oMdes = pin.SE3(np.eye(3), np.array([1., 0., 1.]))

q= np.array([
        0, 0, 0.840252, 0, 0, 0, 1,  # Free flyer
        0, 0, -0.3490658, 0.6981317, -0.3490658, 0,  # left leg
        0, 0, -0.3490658, 0.6981317, -0.3490658, 0,  # right leg
        0,  # chest
        1.5, 0.6, -0.5, -1.05, -0.4, -0.3, -0.2,  # left arm
        0, 0, 0, 0,  # head
        1.5, -0.6, 0.5, 1.05, -0.4, -0.3, -0.2,  # right arm
    ]).T
eps    = 1e-4
IT_MAX = 1000
DT     = 1e-1
damp   = 1e-12

i=0
while True:
    pin.forwardKinematics(model,data,q)
    dMi = oMdes.actInv(data.oMi[JOINT_ID])
    err = pin.log(dMi).vector
    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break
    J = pin.computeJointJacobian(model,data,q,JOINT_ID)
    v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    q = pin.integrate(model,q,v*DT)

    if not i % 10:
        print('%d: error = %s' % (i, err.T))
    i += 1
if success:
    print("Convergence achieved!")
else:
    print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

print('\nresult: %s' % q.flatten().tolist())
print('\nfinal error: %s' % err.T)




