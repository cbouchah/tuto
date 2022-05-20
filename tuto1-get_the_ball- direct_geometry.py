import pinocchio as pin
import hppfcl
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
import sys
import os
from os.path import dirname, join, abspath
import time
from pinocchio.visualize import MeshcatVisualizer


pinocchio_model_dir = join("/home/cbouchah/devel", "pinocchio/models")
model_path = join(pinocchio_model_dir, "example-robot-data/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "romeo_small.urdf"
urdf_model_path = join(join(model_path, "romeo_description/urdf"), urdf_filename)

model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())


data, collision_data, visual_data = pin.createDatas(model, collision_model, visual_model)



model2 = pin.Model()

geom_model = pin.GeometryModel()
geometries = [hppfcl.Sphere(0.1)]

for i, geom in enumerate(geometries):
    placement = pin.SE3(np.eye(3), np.array([0.33, -0.25, 1.11]))
    geom_obj = pin.GeometryObject("obj{}".format(i), 0, 0, geom, placement)
    color = np.random.uniform(0, 1, 4)
    color[3] = 1
    geom_obj.meshColor = color
    geom_model.addGeometryObject(geom_obj)

viz2 = MeshcatVisualizer(model2, geom_model, geom_model)
viz2.initViewer()
viz2.loadViewerModel()
viz2.display(np.zeros(0))

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(viz2.viewer)
viz.loadViewerModel(color = [1, 2, 1.5, 2.1])



for i in np.arange(1.5, -0.1, -0.003):
    q = np.array([
        0, 0, 0.840252, 0, 0, 0, 1,  # Free flyer
        0, 0, -0.3490658, 0.6981317, -0.3490658, 0,  # left leg
        0, 0, -0.3490658, 0.6981317, -0.3490658, 0,  # right leg
        0,  # chest
        1.5, 0.6, -0.5, -1.05, -0.4, -0.3, -0.2,  # left arm
        0, 0, 0, 0,  # head
        i, -0.6, 0.5, 1.05, -0.4, -0.3, -0.2,  # right arm
    ]).T
    viz.display(q)
    time.sleep(0.001)


q1= np.array([
        0, 0, 0.840252, 0, 0, 0, 1,  # Free flyer
        0, 0, -0.3490658, 0.6981317, -0.3490658, 0,  # left leg
        0, 0, -0.3490658, 0.6981317, -0.3490658, 0,  # right leg
        0,  # chest
        1.5, 0.6, -0.5, -1.05, -0.4, -0.3, -0.2,  # left arm
        0, 0, 0, 0,  # head
        1.5, -0.6, 0.5, 1.05, -0.4, -0.3, -0.2,  # right arm
    ]).T
viz.display(q1)
time.sleep(0.001)
q2= np.array([
        0, 0, 0.840252, 0, 0, 0, 1,  # Free flyer
        0, 0, -0.3490658, 0.6981317, -0.3490658, 0,  # left leg
        0, 0, -0.3490658, 0.6981317, -0.3490658, 0,  # right leg
        0,  # chest
        1.5, 0.6, -0.5, -1.05, -0.4, -0.3, -0.2,  # left arm
        0, 0, 0, 0,  # head
        -0.1, -0.6, 0.5, 1.05, -0.4, -0.3, -0.2,  # right arm
    ]).T

pin.forwardKinematics(model, data, q1)
# idx= data.index('RWristRoll')
# p = data.oMi[31].copy()
p2 = data.oMi[31].translation
# x = np.array(2)
# q = np.matrix(x).T
# x = q.getA()[:, 0]
print(p2)
b= np.array([0, 0, 0])
dist= np.linalg.norm(b-p2)
print(dist)

def cost(x):
    # a, b, c =p2
    return np.linalg.norm(x-p2)

x0 = np.array([1, 1, 1])  # Optimize cost without any constraints in BFGS, with traces.
xopt_bfgs = fmin_bfgs(cost, x0)
print('*** Xopt in BFGS =', xopt_bfgs)



# x = np.array(p)
# q = np.matrix(x).T
# x = q.getA()[:, 0]
# print(x)






pin.forwardKinematics(model,data, q2)

pin.updateGeometryPlacements(model, data, collision_model, collision_data)
pin.updateGeometryPlacements(model, data, visual_model, visual_data)

# Print out the placement of each joint of the kinematic tree
print("\nJoint placements:")
for name, oMi in zip(model.names, data.oMi):
    print("{:<24} : {: .3f} {: .2f} {: .2f}".format( name, *oMi.translation.T.flat ))

# # Print out the placement of each collision geometry object
# print("\nCollision object placements:")
# for k, oMg in enumerate(collision_data.oMg):
#     print(("{:d} : {: .2f} {: .2f} {: .2f}"
#           .format( k, *oMg.translation.T.flat )))
#
# # Print out the placement of each visual geometry object
# print("\nVisual object placements:")
# for k, oMg in enumerate(visual_data.oMg):
#     print(("{:d} : {: .2f} {: .2f} {: .2f}"
#           .format( k, *oMg.translation.T.flat )))













# a =np.linspace(q1,q2)
# print(a)

# q[31]=90
# q[32]=90
# q[33]=90
# q[34]=
# q[35]=np.pi/2
# q[36]=np.pi/2
# q[37]=90






print()