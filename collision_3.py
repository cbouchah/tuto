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
index_LHand = robot.model.getJointId("arm_left_5_joint")
index_LShoulderPitch = robot.model.getJointId("arm_left_1_joint")
index_RHand= robot.model.getJointId("arm_right_5_joint")
index_RShoulderPitch = robot.model.getJointId("arm_right_1_joint")

intial_pos_LH = robot.data.oMi[index_LHand].translation
intial_pos_RH = robot.data.oMi[index_RHand].translation

# q0 = pin.neutral(robot.model)
# q0 = np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,
#         0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
#         0.  ,  0.  ,  0.  , -0.75,  0.2 ,  0.  , -1 ,  0.  , -1,
#         0.  ,  0.  ,  0.75, -0.2 ,  0.  , -1.4  ,  0.  ,  0.  ,  0.  ,
#         0.  ,  0.  ,  0.  ])
# q = np.array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#         0.        ,  1.        ,  0.        ,  0.        ,  0.        ,
#         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#         0.        , -1.10024004, -0.05,  0.26812475, -1.8,
#         0      , 0       ,  0.        ,  0.        ,  1.01801519,
#        -0.29259708,  0.55107464, -1.27004112,  0.        ,  0.        ,
#         0.        ,  0.        ,  0.        ,  0.        ])


# pin.forwardKinematics(robot.model,robot.data, q0)
# pin.updateGeometryPlacements(robot.model,robot.data, robot.collision_model, robot.collision_data)
# pin.updateGeometryPlacements(robot.model, robot.data, robot.visual_model, robot.visual_data)
#
# # Print out the placement of each joint of the kinematic tree
# print("\nJoint placements:")
# for name, oMi in zip(robot.model.names, robot.data.oMi):
#     print("{:<24} : {: .3f} {: .2f} {: .2f}".format( name, *oMi.translation.T.flat ))

# viz.display(q)
#
# def cost(q):
#     qtot = np.concatenate((q0[:index_LShoulderPitch+5], q), axis=None)
#     pin.forwardKinematics(robot.model, robot.data, qtot)
#     pos1 = robot.data.oMi[index_LHand].translation
#     # pos2 = robot.data.oMi[index_RHand].translation
#     pos2 = np.array([0.4,0,0.2])
#     return np.linalg.norm(pos1-pos2)
# #function to display q
# # def disp(x):
# #     time.sleep(1)
# #     xtot = np.concatenate((q0[:index_LShoulderPitch+5], x), axis=None)
# #     viz.display(xtot)
# #     print(xtot)
#
# x0 = q0[index_LShoulderPitch+5 : ]
# # x0 = np.zeros(robot.model.nq-index_LShoulderPitch-5)
# xopt_bfgs = fmin_bfgs(cost, x0)
# print(xopt_bfgs)

# q_desired = np.concatenate((q0[:index_LShoulderPitch+5], xopt_bfgs), axis=None)

# q_desired = np.array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#         0.        ,  1.        ,  0.        ,  0.        ,  0.        ,
#         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#         0.        , -1.10024004, -0.23461898,  0.26812475, -1.19042936,
#         0      , 0       ,  0.        ,  0.        ,  1.01801519,
#        -0.29259708,  0.55107464, -1.27004112,  0.        ,  0.        ,
#         0.        ,  0.        ,  0.        ,  0.        ])
q_desired = np.array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  1.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        , -1.27926534,  1.00620022,  0.38333559, -0.95,
        0.        , 0       ,  0.        ,  0.        ,  1.27926534,  -1.00620022,  -0.38333559, -0.83,  0.,0,
        0.        ,  0.        ,  0.        ,  0.        ])
q = np.array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  1.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        , -1.27926534,  1.00620022,  0.38333559, -0.75,
        0.        , 0       ,  0.        ,  0.        ,  1.27926534,  -1.00620022,  -0.38333559, -0.83,  0.,0,
        0.        ,  0.        ,  0.        ,  0.        ])
# a_final = np.array([ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00, -2.41003507e-03,  6.68919817e-03,
#        -9.14805484e-02, -3.84795126e-02, -2.42629672e-03,  8.59099147e-05,
#         5.60523494e-03, -8.30149649e-03, -1.80712333e-01, -7.23291398e-02,
#        -4.47047410e-03, -8.62682257e-05, -1.54948607e-02,  3.08046914e-01,
#        -7.07922719e-03, -5.45883771e-02, -2.75204139e-02,  3.97075863e-02,
#         9.97159456e-04, -3.96773038e-03,  7.94528681e-03,  9.98790003e-06,
#        -1.83388798e-02,  6.92458024e-02,  2.05172581e-02,  1.31826625e+00,
#        -1.15568680e-03,  2.84451856e-03,  1.96016044e-02, -4.01121299e-05,
#         7.04781400e-03, -4.93093663e-05])
viz.display(q)
time.sleep(1)

v = np.zeros(robot.model.nv)
kp = 110
kv = 2 * np.sqrt(kp)
dt = 1e-4
print()
col = CollisionWrapper(robot, viz)
# print(col.computeCollisions(q_desired))
# print(col.getCollisionList())
# for i in range(col.gmodel.ngeoms):
#     print(f"{i} : {col.gmodel.geometryObjects[i].name} ")
i=0
while True:
    torq = -kp *  (q[7:] - q_desired[7:]) - kv * v[6:]
    a_root_not_zero = pin.aba(robot.model, robot.data, q, v, np.concatenate((np.zeros(6,), torq), axis=None))
    a_without_constraints = np.concatenate((np.zeros(6,), a_root_not_zero[6:]), axis=None)
    # print(col.computeCollisions(q))
    # print(a_without_constraints)
    col.computeCollisions(q)
    collisions = col.getCollisionList()
    if len(collisions) == 0 :
        a = a_without_constraints
        # print(a_without_constraints)
    else :
        # col.computeCollisions(q)
        print(collisions)
        # dist = col.getCollisionDistances(collisions)
        # print(dist)
        J = -col.getCollisionJacobian(collisions)
        M = pin.crba(robot.model, robot.data, q)
        a_collision_with_constraints = qpsolvers.solve_ls(np.identity(38), a_without_constraints, J, np.zeros(J.__len__()), W=M, verbose=True)
        a = a_collision_with_constraints
    try :
        v += a* dt
    except :
        print()
    q = pin.integrate(robot.model, q, v * dt)
    # print(q)
    viz.display(q)
    # time.sleep(0.000000001)


# q = np.array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#         0.        ,  1.        ,  0.        ,  0.        ,  0.        ,
#         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#         0.        , -1.27926534,  0.49620022,  0.38333559, -1.81466985,
#         0.        , 0       ,  0.        ,  0.        ,  1.27926534,  -0.49620022,  -0.38333559, -1.81466985,  0.,0,
#         0.        ,  0.        ,  0.        ,  0.        ])
# print()
# while True :
#     c = col.computeCollisions(q)
#     print(c)
#     collisions = col.getCollisionList()
#     # print(collisions)
#     # dist = col.getCollisionDistances(collisions)
#     # print(dist)
#     J = -col.getCollisionJacobian(collisions)
#     # M = pin.crba(robot.model, robot.data, q)
#     a_collision_with_constraints = qpsolvers.solve_ls(np.identity(38), a_without_constraints, J, -np.ones(J.__len__()))
#     v += a_without_constraints * dt
#     q = pin.integrate(robot.model, q, v * dt)
#
#     viz.display(q)
#     print(q)
# print()