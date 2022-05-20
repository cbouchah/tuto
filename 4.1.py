from __future__ import print_function
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from numpy.linalg import norm, solve
from os.path import dirname, join, abspath
import pinocchio as pin
import time
import example_robot_data as robex
from pinocchio.utils import rotate, zero, eye


robot = robex.loadTalosArm()
Viewer = pin.visualize.MeshcatVisualizer
viz = Viewer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True)
# print(viz.display(robot.q0))
# print(robot.q0)
# q=robot.q0
# pin.forwardKinematics(robot.model,robot.data, q)
# Print out the placement of each joint of the kinematic tree
# print("\nJoint placements:")
# for name, oMi in zip(robot.model.names, robot.data.oMi):
#     print("{:<24} : {: .3f} {: .2f} {: .2f}".format( name, *oMi.translation.T.flat ))
q= np.random.rand(robot.model.nq)
vq= np.random.rand(robot.model.nv)
aq0= zero(robot.model.nv)

# compute dynamic drift -- Coriolis, centrifugal, gravity
b = pin.rnea(robot.model, robot.data, q, vq, aq0)
# compute mass matrix M
M = pin.crba(robot.model, robot.data, q)

print()