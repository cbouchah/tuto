from __future__ import print_function
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from numpy.linalg import norm, pinv
from os.path import dirname, join, abspath
import pinocchio as pin
import time
from scipy.optimize import fmin_bfgs, fmin_slsqp

import example_robot_data as robex
from pinocchio.utils import rotate, zero, eye
from utils import *
import hppfcl

robot = robex.loadTalosArm()
Viewer = pin.visualize.MeshcatVisualizer
viz = Viewer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True)
time.sleep(2)


geom_model = pin.GeometryModel()

geometries = [hppfcl.Sphere(0.1)]
for i, geom in enumerate(geometries):
    placement = pin.SE3(np.eye(3), np.array([0.33, -0.25, 1.11]))
    geom_obj = pin.GeometryObject("obj{}".format(i), 0, 0, geom, placement)
    color = np.random.uniform(0, 1, 4)
    color[3] = 1
    geom_obj.meshColor = color
    geom_model.addGeometryObject(geom_obj)
geom_data = geom_model.createData()



q= pin.neutral(robot.model)
# viz.display(q)
v = np.zeros(robot.model.nv)
a = np.zeros(robot.model.nv)
dt = 1e-4
torq = np.zeros(robot.model.nv)
Kp = np.ones(robot.model.nq)*100
Kv = 2 * np.sqrt(Kp)
qdes = np.random.rand(robot.model.nq)*np.pi
viz.display(qdes)
time.sleep(2)


# geom_model.addAllCollisionPairs()
# pin.computeCollisions(robot.model, robot.data, geom_model, geom_data, q, False)
# for k in range(len(geom_model.collisionPairs)):
#     cr = geom_data.collisionResults[k]
#     cp = geom_model.collisionPairs[k]
#     print("collision pair:",cp.first,",",cp.second,"- collision:","Yes" if cr.isCollision() else "No")

time.sleep(2)

while True :
    b = pin.nle(robot.model, robot.data, q, v)
    M = pin.crba(robot.model, robot.data, q)
    torq = -Kp * (q - qdes) - Kv * v
    # torq = pin.rnea(robot.model, robot.data, q, v, a)
    # a = pinv(M) .dot(torq - b)
    a = pin.aba(robot.model, robot.data, q, v, torq)
    v += a * dt

    q = pin.integrate(robot.model, q, v * dt)
    viz.display(q)
    time.sleep(0.0001)
    if  np.linalg.norm(q-qdes)< 1e-4:
        success = True
        break

print()