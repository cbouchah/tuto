from __future__ import print_function
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from numpy.linalg import norm, solve
from os.path import dirname, join, abspath
import pinocchio as pin
import time
from scipy.optimize import fmin_bfgs, fmin_slsqp



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

axi_0= np.array([0, 0, 0, 0, 0, 0, 1])
x0= np.random.rand(38)

def cost(q):
     pin.forwardKinematics(model,data,q)
     frame = q[0:7]
     return np.linalg.norm(axi_0-frame)
def call_back (x):
    time.sleep(1)
    viz.display(x)
    print(x)
call_back(x0)
time.sleep(5)
xopt_bfgs = fmin_bfgs(cost, x0,callback=call_back)
print(xopt_bfgs)

JOINT_ID = 31
desired = pin.SE3(np.eye(3), np.array([1, 1, 1])) #desired position
q= xopt_bfgs
eps    = 1e-4
DT     = 1e-1
damp   = 1e-12

i=0
while True:
    qtot=np.concatenate((np.array([0, 0, 0, 0, 0, 0, 0]),q[7:38]),axis=None)
    pin.forwardKinematics(model,data,q)
    current=data.oMi[JOINT_ID]
    err= pin.log(desired.actInv(current)).vector
    if norm(err) < eps:
     success=True
     break
    if i >= 100:
     success = False
     break
    J = pin.computeJointJacobian(model,data,q,JOINT_ID)
    v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    q = pin.integrate(model,qtot,v*DT)
    viz.display(q)
    time.sleep(0.1)
    if not i % 10:
        print('%d: error = %s' % (i, err.T))

    i += 1
print()

