from __future__ import print_function
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from numpy.linalg import norm, pinv
from os.path import dirname, join, abspath
import pinocchio as pin
import time
import qpsolvers
from scipy.optimize import fmin_bfgs, fmin_slsqp
import example_robot_data as robex
from pinocchio.utils import rotate, zero, eye
from utils_tuto4 import *
import hppfcl
robot = robex.loadTalos(legs=True)
Viewer = pin.visualize.MeshcatVisualizer
viz = Viewer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True)
time.sleep(1)

index_leg_left_1_joint= robot.model.getJointId("leg_left_1_joint")
index_leg_left_6_joint= robot.model.getJointId("leg_left_6_joint")
pos_desired_of_leg_left_6_joint= np.array([0.4   ,  -0.05  , -0.8])

q0 = pin.neutral(robot.model)
#
# pin.forwardKinematics(robot.model,robot.data, q0)
# pin.updateGeometryPlacements(robot.model,robot.data, robot.collision_model, robot.collision_data)
# pin.updateGeometryPlacements(robot.model, robot.data, robot.visual_model, robot.visual_data)
#
# # Print out the placement of each joint of the kinematic tree
# print("\nJoint placements:")
# for name, oMi in zip(robot.model.names, robot.data.oMi):
#     print("{:<24} : {: .3f} {: .2f} {: .2f}".format( name, *oMi.translation.T.flat ))

viz.display(q0)
def cost(q):
    qtot = np.concatenate((q0[:index_leg_left_1_joint+5], q, q0[index_leg_left_6_joint+6:]), axis=None)
    pin.forwardKinematics(robot.model, robot.data, qtot)
    pos_actual_of_leg_left_6_joint = robot.data.oMi[index_leg_left_6_joint].translation
    return np.linalg.norm(pos_actual_of_leg_left_6_joint-pos_desired_of_leg_left_6_joint)
#function to display q
def disp(x):
    time.sleep(1)
    xtot = np.concatenate((q0[:index_leg_left_1_joint], x, q0[index_leg_left_6_joint+1:]), axis=None)
    viz.display(xtot)
    print(xtot)

x0 = np.ones(6)
xopt_bfgs = fmin_bfgs(cost, x0)
print(xopt_bfgs)

q_desired = np.concatenate((q0[:index_leg_left_1_joint], xopt_bfgs, q0[index_leg_left_6_joint+1:]), axis=None)

viz.display(q_desired)
time.sleep(2)
q = pin.neutral(robot.model)
v = np.zeros(robot.model.nv)
kp = 120
kv = 2 * np.sqrt(kp)
dt = 1e-4
print()
# fais la modification pour just cal
while True:
    col = CollisionWrapper(robot, viz)
    torq = -kp *  (q[7:] - q_desired[7:]) - kv * v[6:]
    a_root_not_zero = pin.aba(robot.model, robot.data, q, v, np.concatenate((np.zeros(6,), torq), axis=None))
    a_without_constraints = np.concatenate((np.zeros(6,), a_root_not_zero[6:]), axis=None)
    print(a_without_constraints)
    if col.computeCollisions(q) == False:
        a = a_without_constraints
        # print(a_without_constraints)
    else:
        print(col.computeCollisions(q))
        col.computeCollisions(q)
        collisions = col.getCollisionList()
        # print(collisions)
        dist = col.getCollisionDistances(collisions)
        # print(dist)
        J = -col.getCollisionJacobian(collisions)
        M = pin.crba(robot.model, robot.data, q)
        a_collision_with_constraints = qpsolvers.solve_ls(np.identity(38), a_without_constraints, J, np.zeros(J.__len__()).reshape(J.__len__()))
        a = a_collision_with_constraints
    v += a * dt
    q = pin.integrate(robot.model, q, v * dt)
    print(q)
    viz.display(q)
    time.sleep(0.0001)

print()