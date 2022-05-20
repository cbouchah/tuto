import pinocchio as pin
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from os.path import dirname, join, abspath
from pinocchio.visualize import MeshcatVisualizer
import time
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from pinocchio.explog import log
from pinocchio import SE3


pinocchio_model_dir = join("/home/cbouchah/devel", "pinocchio/models")
model_path = join(pinocchio_model_dir, "example-robot-data/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "romeo_small.urdf"
urdf_model_path = join(join(model_path, "romeo_description/urdf"), urdf_filename)
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())
data, collision_data, visual_data = pin.createDatas(model, collision_model, visual_model)

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer()
viz.loadViewerModel(color = [1, 2, 1.5, 2.1])


q= np.array([
        0, 0, 0.840252, 0, 0, 0, 1,  # Free flyer
        0, 0, -0.3490658, 0.6981317, -0.3490658, 0,  # left leg
        0, 0, -0.3490658, 0.6981317, -0.3490658, 0,  # right leg
        0,  # chest
        1.5, 0.6, -0.5, -1.05, -0.4, -0.3, -0.2,  # left arm
        0, 0, 0, 0,  # head
       1.5 , -0.6, 0.5, 1.05, -0.4, -0.3, -0.2,  # right arm
    ]).T

pin.forwardKinematics(model, data, q)
plac = log(data.oMi[31].copy())
plac_vec=plac.vector
print(plac_vec)

nu = log(SE3.Random())
nu_vec = nu.vector
print(nu_vec)


def cost(desired):
    return np.linalg.norm(desired-plac_vec)
print(plac)

normalization = plac_vec/np.linalg.norm(plac_vec)