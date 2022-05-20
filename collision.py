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
robot = robex.loadTalos()
Viewer = pin.visualize.MeshcatVisualizer
viz = Viewer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True)
time.sleep(1)

index_LHand = robot.model.getJointId("gripper_left_joint")
index_LShoulderPitch = robot.model.getJointId("arm_left_1_joint")
index_RHand= robot.model.getJointId("gripper_right_joint")
index_RShoulderPitch = robot.model.getJointId("arm_right_1_joint")

intial_pos_LH = robot.data.oMi[index_LHand].translation
intial_pos_RH = robot.data.oMi[index_RHand].translation

q0 = pin.neutral(robot.model)
q = np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  , -0.75,  0.2 ,  0.  , -1 ,  0.  , -1,
        0.  ,  0.  ,  0.75, -0.2 ,  0.  , -1.4  ,  0.  ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.  ])
# pin.forwardKinematics(robot.model,robot.data, q0)
# pin.updateGeometryPlacements(robot.model,robot.data, robot.collision_model, robot.collision_data)
# pin.updateGeometryPlacements(robot.model, robot.data, robot.visual_model, robot.visual_data)
#
# # Print out the placement of each joint of the kinematic tree
# print("\nJoint placements:")
# for name, oMi in zip(robot.model.names, robot.data.oMi):
#     print("{:<24} : {: .3f} {: .2f} {: .2f}".format( name, *oMi.translation.T.flat ))


def cost(q):
    qtot = np.concatenate((q0[:index_LShoulderPitch+5], q), axis=None)
    pin.forwardKinematics(robot.model, robot.data, qtot)
    pos1 = robot.data.oMi[index_LHand].translation
    pos2 = robot.data.oMi[index_RHand].translation
    return np.linalg.norm(pos1-pos2)
#function to display q
# def disp(x):
#     time.sleep(1)
#     xtot = np.concatenate((q0[:index_LShoulderPitch+5], x), axis=None)
#     viz.display(xtot)
#     print(xtot)

x0 = q[index_LShoulderPitch+5 : ]
# x0 = np.zeros(robot.model.nq-index_LShoulderPitch-5)
xopt_bfgs = fmin_bfgs(cost, x0)
print(xopt_bfgs)

q_desired = np.concatenate((q0[:index_LShoulderPitch+5], xopt_bfgs), axis=None)

viz.display(q_desired)
time.sleep(2)
v = np.zeros(robot.model.nv)
kp = 110
kv = 2 * np.sqrt(kp)
dt = 1e-4
print()
col = CollisionWrapper(robot, viz)
a_without_constraints = np.array([ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00, -8.93123764e-02, -3.66198214e-01,
        6.59253029e-02,  9.82927657e-02,  1.31373713e-02, -7.44316819e-03,
       -5.91105802e-02, -3.69136920e-01, -1.39680624e-02, -2.82693895e-02,
       -3.50547816e-03, -7.68674598e-03,  2.44933745e-01,  3.34826641e-01,
       -1.55378468e-02,  9.11471861e-02, -9.61764333e-01,  3.13485173e+00,
       -8.43211997e-01, -9.57760228e-02,  1.12942697e+00, -1.01213644e-03,
        6.33656838e-01,  8.37793882e-01, -3.46749492e-01, -3.09283798e+00,
        8.21254778e-02, -2.87653210e-01, -9.15608340e-01, -7.48788740e-03,
        5.71705160e-03, -1.27841943e-03])
# fais la modification pour just cal
while True:
    # torq = -kp *  (q[7:] - q_desired[7:]) - kv * v[6:]
    # a_root_not_zero = pin.aba(robot.model, robot.data, q, v, np.concatenate((np.zeros(6,), torq), axis=None))
    # a_without_constraints = np.concatenate((np.zeros(6,), a_root_not_zero[6:]), axis=None)
    # # # print(a_without_constraints)
    # if col.computeCollisions(q) == True:
    #     break
    #     a = a_without_constraints
    #     # print(a_without_constraints)
    # else:
    # print(col.computeCollisions(q))
    col.computeCollisions(q_desired)
    collisions = col.getCollisionList()
    # print(collisions)
    # dist = col.getCollisionDistances(collisions)
    # print(dist)
    J = -col.getCollisionJacobian(collisions)
    # M = pin.crba(robot.model, robot.data, q)
    a_collision_with_constraints = qpsolvers.solve_ls(np.identity(38), a_without_constraints, J, np.zeros(J.__len__()).reshape(J.__len__()))
    a = a_collision_with_constraints
    v += a * dt
    q = pin.integrate(robot.model, q, v * dt)
    print(q)
    viz.display(q)
    # time.sleep(0.000000001)


print()