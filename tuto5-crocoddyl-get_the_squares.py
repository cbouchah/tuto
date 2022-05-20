import time

import crocoddyl
import crocoddyl
import pinocchio
import numpy as np
import example_robot_data as robex

robot = robex.load('talos_arm')
robot_model = robot.model
robot_model.armature =np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.])*5
robot_model.q0 = np.array([3.5,2,2,0,0,0,0])
robot_model.x0 = np.concatenate([robot_model.q0, np.zeros(robot_model.nv)])
robot_model.gravity *= 0

from visualizer_utils import MeshcatVisualizer
viz = MeshcatVisualizer(robot)
viz.display(robot_model.q0)
viz.viewer.jupyter_cell()

viz.addBox('world/goal1',[.1,.1,.1], [1.,0,0,1])
viz.addBox('world/goal2',[.1,.1,.1],[0,1,0,1])
viz.addBox('world/goal3',[.1,.1,.1],[1,1,0,1])
viz.addBox('world/goal4',[.1,.1,.1],[1,0,1,1])
viz.applyConfiguration('world/goal1',[0.4,0,0,0,0,0,1])
viz.applyConfiguration('world/goal2',[0.4,0,0.4,0,0,0,1])
viz.applyConfiguration('world/goal3',[0.4,0.4,0,0,0,0,1])
viz.applyConfiguration('world/goal4',[0.4,0.4,0.4,0,0,0,1])


FRAME_TIP = robot_model.getFrameId("gripper_left_fingertip_3_link")
goal_1 = np.array([0.4,0,0])
goal_2 = np.array([0.4,0,0.4])
goal_3 = np.array([0.4,0.4,0])
goal_4 = np.array([0.4,0.4,0.4])

state_1 = crocoddyl.StateMultibody(robot_model)
runningCostModel_1 = crocoddyl.CostModelSum(state_1)
terminalCostModel_1 = crocoddyl.CostModelSum(state_1)

state_2 = crocoddyl.StateMultibody(robot_model)
runningCostModel_2 = crocoddyl.CostModelSum(state_2)
terminalCostModel_2 = crocoddyl.CostModelSum(state_2)

state_3 = crocoddyl.StateMultibody(robot_model)
runningCostModel_3 = crocoddyl.CostModelSum(state_3)
terminalCostModel_3 = crocoddyl.CostModelSum(state_3)

state_4 = crocoddyl.StateMultibody(robot_model)
runningCostModel_ = crocoddyl.CostModelSum(state_4)
terminalCostModel_4 = crocoddyl.CostModelSum(state_4)

# for i in range(1,4):
#     ### Cost for reaching the target
#     # Mref = crocoddyl.FramePlacement(FRAME_TIP,pinocchio.SE3(np.eye(3), goal_))
#     pref_i = crocoddyl.FrameTranslation(FRAME_TIP,goal_i)
#     goalTrackingCost_i = crocoddyl.CostModelFrameTranslation(state_i, pref_i)
#     weights_i = crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1,1,1,1,1,1,1,1,1,2,2,2.]))
#
#     ### Cost for regularizing the state about robot_model.x0
#     xRegCost_i = crocoddyl.CostModelState(state_i,weights_i,robot_model.x0)
#     weightsT_i =crocoddyl.ActivationModelWeightedQuad(np.array([.01,.01,.01,.01,.01,.01,.01, 1,1,1,1,2,2,2.]))
#     xRegCostT_i = crocoddyl.CostModelState(state_1,weightsT_i,robot_model.x0)
#
#     ### Cost for keeping the control low
#     uRegCost_i = crocoddyl.CostModelControl(state_i)
#
#     runningCostModel_i.addCost("gripperPose", goalTrackingCost_i, .001)
#     runningCostModel_i.addCost("xReg", xRegCost_i, 1e-3)
#     runningCostModel_i.addCost("uReg", uRegCost_i, 1e-6)
#     terminalCostModel_i.addCost("gripperPose", goalTrackingCost_i, 10)
#     terminalCostModel_i.addCost("xReg", xRegCostT_i, .01)

### Cost for reaching the target
# Mref = crocoddyl.FramePlacement(FRAME_TIP,pinocchio.SE3(np.eye(3), goal_))
pref_1 = crocoddyl.FrameTranslation(FRAME_TIP,goal_1)
goalTrackingCost_1 = crocoddyl.CostModelFrameTranslation(state_1, pref_1)
weights_1=crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1,1,1,1,1,1,1,1,1,2,2,2.]))

### Cost for regularizing the state about robot_model.x0
xRegCost_1 = crocoddyl.CostModelState(state_1 ,weights_1 ,robot_model.x0)
weightsT_1 =crocoddyl.ActivationModelWeightedQuad(np.array([.01,.01,.01,.01,.01,.01,.01, 1,1,1,1,2,2,2.]))
xRegCostT_1 = crocoddyl.CostModelState(state_1 ,weightsT_1 ,robot_model.x0)

### Cost for keeping the control low
uRegCost_1 = crocoddyl.CostModelControl(state_1 )

runningCostModel_1.addCost("gripperPose", goalTrackingCost_1 , .001)
runningCostModel_1.addCost("xReg", xRegCost_1 , 1e-3)
runningCostModel_1.addCost("uReg", uRegCost_1 , 1e-6)
terminalCostModel_1.addCost("gripperPose", goalTrackingCost_1 , 10)
terminalCostModel_1.addCost("xReg", xRegCostT_1 , .01)

### Cost for reaching the target
# Mref = crocoddyl.FramePlacement(FRAME_TIP,pinocchio.SE3(np.eye(3), goal_))
pref_2= crocoddyl.FrameTranslation(FRAME_TIP,goal_2)
goalTrackingCost_2 = crocoddyl.CostModelFrameTranslation(state_2, pref_2)
weights_2 =crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1,1,1,1,1,1,1,1,1,2,2,2.]))

### Cost for regularizing the state about robot_model.x0
xRegCost_2 = crocoddyl.CostModelState(state_2 ,weights_2 ,robot_model.x0)
weightsT_2 =crocoddyl.ActivationModelWeightedQuad(np.array([.01,.01,.01,.01,.01,.01,.01, 1,1,1,1,2,2,2.]))
xRegCostT_2 = crocoddyl.CostModelState(state_2 ,weightsT_2 ,robot_model.x0)

### Cost for keeping the control low
uRegCost_2 = crocoddyl.CostModelControl(state_2)

runningCostModel_2.addCost("gripperPose", goalTrackingCost_2, .001)
runningCostModel_2.addCost("xReg", xRegCost_2, 1e-3)
runningCostModel_2.addCost("uReg", uRegCost_2, 1e-6)
terminalCostModel_2.addCost("gripperPose", goalTrackingCost_2, 10)
terminalCostModel_2.addCost("xReg", xRegCostT_2, .01)

# ### Cost for reaching the target
# # Mref = crocoddyl.FramePlacement(FRAME_TIP,pinocchio.SE3(np.eye(3), goal_))
# pref_= crocoddyl.FrameTranslation(FRAME_TIP,goal_)
# goalTrackingCost_ = crocoddyl.CostModelFrameTranslation(state_, pref_)
# weights_=crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1,1,1,1,1,1,1,1,1,2,2,2.]))
#
# ### Cost for regularizing the state about robot_model.x0
# xRegCost_ = crocoddyl.CostModelState(state_,weights_,robot_model.x0)
# weightsT_=crocoddyl.ActivationModelWeightedQuad(np.array([.01,.01,.01,.01,.01,.01,.01, 1,1,1,1,2,2,2.]))
# xRegCostT_ = crocoddyl.CostModelState(state_,weightsT_,robot_model.x0)
#
# ### Cost for keeping the control low
# uRegCost_ = crocoddyl.CostModelControl(state_)
#
# runningCostModel_.addCost("gripperPose", goalTrackingCost_, .001)
# runningCostModel_.addCost("xReg", xRegCost_, 1e-3)
# runningCostModel_.addCost("uReg", uRegCost_, 1e-6)
# terminalCostModel_.addCost("gripperPose", goalTrackingCost_, 10)
# terminalCostModel_.addCost("xReg", xRegCostT_, .01)
#





dt = 1e-2
actuationModel_1 = crocoddyl.ActuationModelFull(state_1)
runningModel_1 = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state_1, actuationModel_1, runningCostModel_1), dt)
runningModel_1.differential.armature = robot_model.armature
terminalModel_1 = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state_1, actuationModel_1, terminalCostModel_1), 0.)
terminalModel_1.differential.armature = robot_model.armature

actuationModel_2 = crocoddyl.ActuationModelFull(state_2)
runningModel_2 = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state_2, actuationModel_2, runningCostModel_2), dt)
runningModel_2.differential.armature = robot_model.armature
terminalModel_2 = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state_2, actuationModel_2, terminalCostModel_2), 0.)
terminalModel_2.differential.armature = robot_model.armature

T = 100
seq0 = [runningModel_1]*T + [terminalModel_1]
seq1 = [runningModel_2]*T
problem = crocoddyl.ShootingProblem(robot_model.x0, seq0+seq1, terminalModel_2)

ddp = crocoddyl.SolverDDP(problem)

ddp.setCallbacks([crocoddyl.CallbackLogger(),crocoddyl.CallbackVerbose()])

ddp.solve([],[],1000)  # xs_init,us_init,maxiter
# %load -r 103-104 tp5/arm_example.py
import tuto_5_utils as crocutils
for i in range(4):
    crocutils.displayTrajectory(viz,ddp.xs,ddp.problem.runningModels[i].dt,12)
    time.sleep(1)
    print(robot.q0)

# %load -r 97-99 tp5/arm_example.py
log = ddp.getCallbacks()[0]
crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1)

crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)


dataCollector = crocoddyl.DataCollectorMultibody(robot.data)
trackData = goalTrackingCost.createData(dataCollector)

x = ddp.xs[1].copy()
q = x[:state.nq]
pinocchio.updateFramePlacements(robot.model, robot.data)
pinocchio.computeJointJacobians(robot.model, robot.data, q)
goalTrackingCost.calc(trackData, x)
goalTrackingCost.calcDiff(trackData, x)
print('Lx = ',trackData.Lx)
print('Lu = ',trackData.Lu)

