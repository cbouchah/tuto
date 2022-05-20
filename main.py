# This is a sample Python script.
import numpy as np
import pinocchio as pin
import sys
import os
from os.path import dirname, join, abspath
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    A = np.matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
    b = np.zeros([4, 1])
    c = A * b

R = pin.utils.eye(3)
p = pin.utils.zero(3)
M0 = pin.SE3(R, p)
M = pin.SE3.Random()
M.translation = p
M.rotation = R

v = pin.utils.zero(3)
w = pin.utils.zero(3)
nu0 = pin.Motion(v, w)
nu = pin.Motion.Random()
nu.linear = v
nu.angular = w

f = pin.utils.zero(3)
tau = pin.utils.zero(3)
phi0 = pin.Force(f, tau)
phi = pin.Force.Random()
phi.linear = f
phi.angular = tau


model_dir = "/home/cbouchah/devel/pinocchio/models"
urdf_filename = "example-robot-data/robots/ur_description/urdf/ur5_robot.urdf"
urdf_model_path = join(model_dir, urdf_filename)
robot = pin.robot_wrapper.buildModelsFromUrdf(urdf_model_path, model_dir)

print(robot)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/


