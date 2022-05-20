import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import  MeshcatVisualizer
from os.path import dirname, join, abspath

pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "pinocchio/models")

model_path = join(pinocchio_model_dir,"example-robot-data/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "talos_reduced_box.urdf"
urdf_model_path = join(join(model_path,"talos_data/robots"),urdf_filename)

robot = RobotWrapper.BuildFromURDF(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())

# model = robot.model
# data = robot.data
q0 = robot.q0


robot.setVisualizer(MeshcatVisualizer)
robot.initViewer()
robot.loadViewerModel("pinocchio")
robot.display(q0)
print()