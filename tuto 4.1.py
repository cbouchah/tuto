from __future__ import print_function
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from numpy.linalg import norm, pinv
from os.path import dirname, join, abspath
import pinocchio as pin
import time
from utils import display_sphere
import example_robot_data as robex
from pinocchio.utils import rotate, zero, eye

robot = robex.loadTalosArm()
Viewer = pin.visualize.MeshcatVisualizer
viz = Viewer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True)
time.sleep(2)


q= pin.neutral(robot.model)
viz.display(q)
v= np.random.rand(robot.model.nv)
aq0= zero(robot.model.nv)
DT     = 1e-2
# # compute dynamic drift -- Coriolis, centrifugal, gravity
b = pin.rnea(robot.model, robot.data, q, v, aq0)
# compute mass matrix M
M = pin.crba(robot.model, robot.data, q)
# torq = np.random.rand(7)
Kp = np.array([0.05,0.05,0.05,0.05,0.05,0.05,0.05])
Kv = 2 * np.sqrt(Kp)
qdes = np.array([1,1,1,1,1,1,1])
time.sleep(2)
# torq = -Kp.dot(q-)
# torq=np.array([0,0,0,0,0,0,0])
viz.display(qdes)
time.sleep(2)
# a = (torq-b).dot(pinv(M))
while True :
    for i in range(0, 100):
        torq = -Kp*(q-qdes) - Kv*(v)
        a = pinv(M).dot((torq - b))
        # b = pin.rnea(robot.model, robot.data, q, v, a)
        # M = pin.crba(robot.model, robot.data, q)
        v = pin.integrate(robot.model, q, a * DT)
        q = pin.integrate(robot.model, q, v * DT)
        # torq= -np.array([1, 1, 1,1,1,1,1]).dot(v)
        print(q)
        viz.display(q)
        time.sleep(0.1)
        if norm(q-qdes) < 1e-4:
            success = True
            break

print()