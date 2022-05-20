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



robot = robex.loadTalosArm()
Viewer = pin.visualize.MeshcatVisualizer
viz = Viewer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True)
time.sleep(2)
q = np.array([0, 0, 0.2, 0, 0, 0.3,0])*np.pi
qdes = np.array([0, 0, 0.5, 0, 0, 0.8,0])*np.pi
v = np.zeros(robot.model.nv)
a = np.zeros(robot.model.nv)
dt = 1e-4
torq = np.zeros(robot.model.nv)
Kp = 110
Kv = 2 * np.sqrt(Kp)
# qdes =np.array([1,-3.14,1,-3.14,3.14,-3.14,3.14])
# qdes = np.random.rand(robot.model.nq)*np.pi
viz.display(qdes)
time.sleep(2)
print()
# while True:
#     b = pin.nle(robot.model, robot.data, q, v)
#     M = pin.crba(robot.model, robot.data, q)
#     a = pin.aba(robot.model, robot.data, q, v, torq)
#     v += a * dt
#     q = pin.integrate(robot.model, q, v * dt)
#     viz.display(q)
#     print(a)
#     time.sleep(0.0001)

# robot.collision_model.addAllCollisionPairs()
# coll_data = robot.collision_model.createData()
# dist = pin.computeCollisions(robot.model, robot.data, robot.collision_model, coll_data , q, False)
# for k in range(len(robot.collision_model.collisionPairs)):
#   cr = coll_data.collisionResults[k]
#   cp = robot.collision_model.collisionPairs[k]
#   print("collision pair:",cp.first,",",cp.second,"- collision:","Yes" if cr.isCollision() else "No")
#
# collision_index = 0
# contact = coll_data.collisionResults[collision_index].getContact(0)
# oMc = pin.SE3(pin.Quaternion.FromTwoVectors(np.array([0,0,1]),contact.normal).matrix(),contact.pos)
# joint_1 = 0
# joint_2 = 1
# oMj1 = robot.data.oMi[joint_1]
# oMj2 = robot.data.oMi[joint_2]
# cMj1 = oMc.inverse()*oMj1
# cMj2 = oMc.inverse()*oMj2
# J1 = robot.computeJointJacobian(q, joint_1)
# J2 = robot.computeJointJacobian(q, joint_2)
# Jc1 = cMj1.action@J1
# Jc2 = cMj2.action@J2
# J = (Jc1-Jc2)[2,]

# pin.forwardKinematics(robot.model,robot.data, q)
#
# pin.updateGeometryPlacements(robot.model,robot.data, robot.collision_model, coll_data)
# pin.updateGeometryPlacements(robot.model, robot.data, robot.visual_model, robot.visual_data)
#
# # Print out the placement of each joint of the kinematic tree
# print("\nJoint placements:")
# for name, oMi in zip(robot.model.names, robot.data.oMi):
#     print("{:<24} : {: .3f} {: .2f} {: .2f}".format( name, *oMi.translation.T.flat ))

# b = pin.nle(robot.model, robot.data, q, v)
# M = pin.crba(robot.model, robot.data, q)
print()
while True :
    col = CollisionWrapper(robot, viz)
    print(col.computeCollisions(q))
    torq = -Kp * (q - qdes) - Kv * v
    # b = pin.nle(robot.model, robot.data, q, v)
    # M = pin.crba(robot.model, robot.data, q)
    # a_free = pinv(M) .dot(torq - b)
    a_free = pin.aba(robot.model, robot.data, q, v, torq)
    if col.computeCollisions(q) == False:
        a = a_free
    else:
        col.computeCollisions(q)
        collisions = col.getCollisionList()
        print(collisions)
        dist = col.getCollisionDistances(collisions)
        print(dist)
        J = -col.getCollisionJacobian(collisions)
        M = pin.crba(robot.model, robot.data, q)
        a_collision = qpsolvers.solve_ls(np.identity(7), a_free, J, np.zeros(J.__len__()))
        a = a_collision
    v += a * dt
    q = pin.integrate(robot.model, q, v * dt)
    viz.display(q)
    time.sleep(0.00001)

print()