import pinocchio as pin
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from os.path import dirname, join, abspath
from pinocchio.visualize import MeshcatVisualizer
import time
import hppfcl

#steps to load romeo robot
pinocchio_model_dir = join("/home/cbouchah/devel", "pinocchio/models")
model_path = join(pinocchio_model_dir, "example-robot-data/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "romeo_small.urdf"
urdf_model_path = join(join(model_path, "romeo_description/urdf"), urdf_filename)
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())
data, collision_data, visual_data = pin.createDatas(model, collision_model, visual_model)


q= np.array([
        0, 0, 0.840252, 0, 0, 0, 1,  # Free flyer
        0, 0, -0.3490658, 0.6981317, -0.3490658, 0,  # left leg
        0, 0, -0.3490658, 0.6981317, -0.3490658, 0,  # right leg
        0,  # chest
        1.5, 0.6, -0.5, -1.05, -0.4, -0.3, -0.2,  # left arm
        0, 0, 0, 0,  # head
       1.5 , -0.6, 0.5, 1.05, -0.4, -0.3, -0.2,  # right arm
    ]).T


#steps to load the robot and the sphere
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer()
viz.loadViewerModel(color = [1, 2, 1.5, 2.1])
viz.display(q)
print()

def cost(q):
    qtot = np.concatenate((np.array([0, 0, 0.840252, 0, 0, 0, 1]), q), axis=None)
    pin.forwardKinematics(model, data, qtot)
    pos_knee_joint = data.oMi[10].translation
    pos_ankle_joint = data.oMi[11].translation
    pos_knee_joint_desired = np.array([0., -0.096, 0.640212])
    pos_ankle_joint_desired = np.array([0.10944643, -0.096, 0.33951036])
    return np.linalg.norm(pos_ankle_joint - pos_ankle_joint_desired), np.linalg.norm(
        pos_knee_joint - pos_knee_joint_desired)
#cost function inputs (q and desire position) output (distance between the q given and the desire)


#function to display q
def disp(x):
    time.sleep(0.001)
    viz.display(x)
    print(x)

x0 = pin.neutral(model)
disp(x0)
time.sleep(5)
for t in range(10) :
    def cost(q):
        pin.forwardKinematics(model, data, q)
        pos_knee_joint_left = data.oMi[10].translation
        # pos_ankle_joint_left = data.oMi[11].translation
        pos_knee_joint_desired_left = np.array([0 + t, -0.096, 0.640212])
        # pos_ankle_joint_desired_left = np.array([0.10944643 + t, -0.096, 0.33951036])
        return np.linalg.norm(pos_knee_joint_left - pos_knee_joint_desired_left)
    xopt_bfgs = fmin_bfgs(cost, x0,callback=disp, maxiter=10, disp=False)
    print(xopt_bfgs)
print()
