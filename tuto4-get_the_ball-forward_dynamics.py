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
from vizutils_tuto4_addgeometry import *




robot = robex.loadTalosArm()
Viewer = pin.visualize.MeshcatVisualizer
viz = Viewer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True)

addViewerSphere(viz,'world/ball', .05, [.2, .2, 1., .5])
applyViewerConfiguration(viz, 'world/ball', [.2,.2,.2, 1, 0, 0, 0])

q= pin.neutral(robot.model)
v = np.zeros(robot.model.nv)
a = np.zeros(robot.model.nv)
dt = 1e-4
torq = np.zeros(robot.model.nv)
Kp = np.ones(robot.model.nq)*100
Kv = 2 * np.sqrt(Kp)
# qdes = np.random.rand(robot.model.nq)*np.pi
# qdes =np.array([1,-3.14,1,-3.14,3.14,-3.14,3.14])
# qdes = np.random.rand(robot.model.nq)*np.pi
# viz = MeshcatVisualizer(robot)
# viz.display(q)
# viz.viewer.jupyter_cell()
# viz.addBox('world/goal',[.1, .1, .1], [1., 0, 0, 1])
# viz.applyConfiguration('world/goal',[0.4,0.4,0.4, 0, 0, 0, 1])
# time.sleep(2)

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
# joint_2 = 2
# oMj1 = robot.data.oMi[joint_1]
# oMj2 = robot.data.oMi[joint_2]
# cMj1 = oMc.inverse()*oMj1
# cMj2 = oMc.inverse()*oMj2
# J1 = robot.computeJointJacobian(q, joint_1)
# J2 = robot.computeJointJacobian(q, joint_2)
# Jc1 = cMj1.action@J1
# Jc2 = cMj2.action@J2
# J = (Jc1-Jc2)

# pin.forwardKinematics(robot.model,robot.data, q)
# pin.updateFramePlacements(robot.model, robot.data)
# pin.updateGeometryPlacements(robot.model,robot.data, robot.collision_model, robot.collision_data)
# pin.updateGeometryPlacements(robot.model, robot.data, robot.visual_model, robot.visual_data)
#
# # Print out the placement of each joint of the kinematic tree
# print("\nJoint placements:")
# for name, oMi in zip(robot.model.names, robot.data.oMi):
#     print("{:<24} : {: .3f} {: .2f} {: .2f}".format( name, *oMi.translation.T.flat ))

b = pin.nle(robot.model, robot.data, q, v)
M = pin.crba(robot.model, robot.data, q)
a_free = pinv(M).dot(np.zeros(robot.model.nv)-b)
position_hand_desired = pin.SE3(np.eye(3),np.array([.2,.2,.2]))

tool_frame_id = robot.model.getFrameId("gripper_left_joint")
i=0

while True:
    col  = CollisionWrapper(robot,viz)
    print(col.computeCollisions(q))
    if col.computeCollisions(q) == False:
        pin.forwardKinematics(robot.model, robot.data, q)
        pin.updateFramePlacements(robot.model, robot.data)
        position_hand = robot.data.oMf[tool_frame_id]
        err = pin.log(position_hand.actInv(position_hand_desired)).vector
        if norm(err) < 1e-4:
            success = True
            break
        if i >= 10000:
            success = False
            break
        J1 = pin.computeFrameJacobian(robot.model, robot.data, q, tool_frame_id)
        vq = pinv(J1).dot(err)
        q = pin.integrate(robot.model, q, vq * 1e-2)
        if not i % 10:
            print('%d: error1 = %s' % (i, err.T))
        viz.display(q)
    else:
        P1 = np.identity(7) - pinv(J1).dot(J1)
        pin.forwardKinematics(robot.model, robot.data, q)
        pin.updateFramePlacements(robot.model, robot.data)
        position_hand = robot.data.oMf[tool_frame_id]
        err = pin.log(position_hand.actInv(position_hand_desired)).vector
        col.computeCollisions(q)
        collisions = col.getCollisionList()
        dist = col.getCollisionDistances(collisions)
        if norm(err) < 1e-4:
            success = True
            break
        J2 = col.getCollisionJacobian(collisions)
        # a = qpsolvers.solve_ls(np.identity(7), a_free, J2, np.zeros(J2.__len__()).reshape(J2.__len__()),W=M)
        vq = P1.dot(pinv(J2).dot(err))
        # vq2 += a * dt
        # vq = vq - pinv(J2).dot(J2).dot(vq)
        q = pin.integrate(robot.model, q, vq * 1e-3)
        print(q)
        viz.display(q)
    i += 1

print()