import time

import pinocchio as pin
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
import hppfcl


def display_sphere(des_SE3):
    model2 = pin.Model()
    geom_model = pin.GeometryModel()
    geometries = [hppfcl.Sphere(0.1)]
    for i, geom in enumerate(geometries):
        geom_obj = pin.GeometryObject("obj{}".format(i), 0, 0, geom, des_SE3)
        color = np.random.uniform(0, 1, 4)
        color[3] = 1
        geom_obj.meshColor = color
        geom_model.addGeometryObject(geom_obj)
    viz2 = MeshcatVisualizer(model2, geom_model, geom_model)
    viz2.initViewer()
    viz2.loadViewerModel()
    viz2.display(np.zeros(0))
    return viz2

def capsule_cylinder(name,joint,radius,length,placement,color=[.7,.7,0.98,1]):
    '''Create a Pinocchio::FCL::Capsule to be added in the Geom-Model. '''
    ### They should be capsules ... but hppfcl current version is buggy with Capsules...
    #hppgeom = hppfcl.Capsule(radius,length)
    hppgeom = hppfcl.Cylinder(radius,length)
    geom = pin.GeometryObject(name,joint,hppgeom,placement)
    geom.meshColor = np.array(color)
    return geom


class CollisionWrapper:
    def __init__(self, robot, viz=None):
        self.robot = robot
        self.viz = viz

        self.rmodel = robot.model
        self.rdata = self.rmodel.createData()
        self.gmodel = self.robot.collision_model
        self.gdata = self.gmodel.createData()
        self.gdata.collisionRequests.enable_contact = True

    def computeCollisions(self, q, vq=None):
        res = pin.computeCollisions(self.rmodel, self.rdata, self.gmodel, self.gdata, q, False)
        # pin.computeDistances(self.rmodel, self.rdata, self.gmodel, self.gdata, q)
        # pin.computeJointJacobians(self.rmodel, self.rdata, q)
        if vq is not None:
            pin.forwardKinematics(self.rmodel, self.rdata, q, vq, 0 * vq)
        return res

    def getCollisionList(self):
        '''Return a list of triplets [ index,collision,result ] where index is the
        index of the collision pair, colision is gmodel.collisionPairs[index]
        and result is gdata.collisionResults[index].
        '''
        return [[ir, self.gmodel.collisionPairs[ir], r]
                for ir, r in enumerate(self.gdata.collisionResults) if r.isCollision()]

    def _getCollisionJacobian(self, col, res):
        '''Compute the jacobian for one collision only. '''
        contact = res.getContact(0)
        g1 = self.gmodel.geometryObjects[col.first]
        g2 = self.gmodel.geometryObjects[col.second]
        oMc = pin.SE3(pin.Quaternion.FromTwoVectors(np.array([0, 0, 1]), contact.normal).matrix(), contact.pos)

        joint1 = g1.parentJoint
        joint2 = g2.parentJoint
        oMj1 = self.rdata.oMi[joint1]
        oMj2 = self.rdata.oMi[joint2]

        cMj1 = oMc.inverse() * oMj1
        cMj2 = oMc.inverse() * oMj2

        J1 = pin.getJointJacobian(self.rmodel, self.rdata, joint1, pin.ReferenceFrame.LOCAL)
        J2 = pin.getJointJacobian(self.rmodel, self.rdata, joint2, pin.ReferenceFrame.LOCAL)
        Jc1 = cMj1.action @ J1
        Jc2 = cMj2.action @ J2
        J = (Jc1 - Jc2)[2,:]
        return J

    def _getCollisionJdotQdot(self, col, res):
        '''Compute the Coriolis acceleration for one collision only. '''
        contact = res.getContact(0)
        g1 = self.gmodel.geometryObjects[col.first]
        g2 = self.gmodel.geometryObjects[col.second]
        oMc = pin.SE3(pin.Quaternion.FromTwoVectors(np.array([0, 0, 1]), contact.normal).matrix(), contact.pos)

        joint1 = g1.parentJoint
        joint2 = g2.parentJoint
        oMj1 = self.rdata.oMi[joint1]
        oMj2 = self.rdata.oMi[joint2]

        cMj1 = oMc.inverse() * oMj1
        cMj2 = oMc.inverse() * oMj2

        a1 = self.rdata.a[joint1]
        a2 = self.rdata.a[joint2]
        a = (cMj1 * a1 - cMj2 * a2).linear[2]
        return a

    def getCollisionJacobian(self, collisions=None):
        '''From a collision list, return the Jacobian corresponding to the normal direction.  '''
        if collisions is None: collisions = self.getCollisionList()
        if len(collisions) == 0: return np.ndarray([0, self.rmodel.nv])
        J = np.vstack([self._getCollisionJacobian(c, r) for (i, c, r) in collisions])
        return J

    def getCollisionJdotQdot(self, collisions=None):
        if collisions is None: collisions = self.getCollisionList()
        if len(collisions) == 0: return np.array([])
        a0 = np.vstack([self._getCollisionJdotQdot(c, r) for (i, c, r) in collisions])
        return a0

    def getCollisionDistances(self, collisions=None):
        if collisions is None: collisions = self.getCollisionList()
        if len(collisions) == 0: return np.array([])
        # dist = np.array([ self.gdata.distanceResults[i].min_distance for (i,c,r) in collisions ])
        dist = []
        for i in range (self.gdata.distanceResults.__len__()):
            distance = self.gdata.distanceResults[i].min_distance
            if distance < 1e-3 :
                dist.append(distance)
        return np.array(dist)
        # return dist

    # --- DISPLAY -----------------------------------------------------------------------------------
    # --- DISPLAY -----------------------------------------------------------------------------------
    # --- DISPLAY -----------------------------------------------------------------------------------

    def initDisplay(self, viz=None):
        if viz is not None: self.viz = viz
        assert (self.viz is not None)

        self.patchName = 'world/contact_%d_%s'
        self.ncollisions = 10
        self.createDisplayPatchs(0)

    def createDisplayPatchs(self, ncollisions):

        if ncollisions == self.ncollisions:
            return
        elif ncollisions < self.ncollisions:  # Remove patches
            for i in range(ncollisions, self.ncollisions):
                self.viz[self.patchName % (i, 'a')].delete()
                self.viz[self.patchName % (i, 'b')].delete()
        else:
            for i in range(self.ncollisions, ncollisions):
                self.viz.addCylinder(self.patchName % (i, 'a'), .0005, .005, "red")
                # viz.addCylinder( self.patchName % (i,'b') , .0005,.05,"red")

        self.ncollisions = ncollisions

    def displayContact(self, ipatch, contact):
        '''
        Display a small red disk at the position of the contact, perpendicular to the
        contact normal.

        @param ipatchf: use patch named "world/contact_%d" % contactRef.
        @param contact: the contact object, taken from Pinocchio (HPP-FCL) e.g.
        geomModel.collisionResults[0].geContact(0).
        '''
        name = self.patchName % (ipatch, 'a')
        R = pin.Quaternion.FromTwoVectors(np.array([0, 1, 0]), contact.normal).matrix()
        M = pin.SE3(R, contact.pos)
        self.viz.applyConfiguration(name, M)

    def displayCollisions(self, collisions=None):
        '''Display in the viewer the collision list get from getCollisionList().'''
        if self.viz is None: return
        if collisions is None: collisions = self.getCollisionList()

        self.createDisplayPatchs(len(collisions))
        for ic, [i, c, r] in enumerate(collisions):
            self.displayContact(ic, r.getContact(0))

# import example_robot_data as robex
# robot = robex.load('talos_arm')
# Viewer = pin.visualize.MeshcatVisualizer
# viz = Viewer(robot.model, robot.collision_model, robot.visual_model)
# viz.initViewer(loadModel=True)
# q=np.array([3.14,-3.14,3.14,-3.14,3.14,-3.14,3.14])
# # viz=viz.display(q)
# a = np.ones(robot.model.nv)
# import qpsolvers
# from numpy.linalg import norm, pinv
#
# col  = CollisionWrapper(robot,viz)
# while True:
#     col.computeCollisions(q)
#     collisions = col.getCollisionList()
#     dist=col.getCollisionDistances(collisions)
#     J = -col.getCollisionJacobian(collisions)
#     a = col.getCollisionJdotQdot(collisions)
    # a = np.array([1,2,3,4,5,6,7])
    # M = np.array([1,2,3,4,5,6,7])
    # v = np.zeros(robot.model.nv)
    # b = pin.nle(robot.model, robot.data, q, v)
    # m = np.ones(49).reshape((7,7))
    # M = pin.crba(robot.model, robot.data, q)
    # a_free = pinv(M).dot(np.zeros(robot.model.nv)-b)
    # a = qpsolvers.solve_ls(M, a_free, J, np.zeros(J.__len__()).reshape(J.__len__(),))
    # a = qpsolvers.solve_ls(np.ones(9).reshape(3,3), a, J, np.zeros(robot.model.nq), W=M)
    # dt = 1e-4
    # v += a * dt
    # q = pin.integrate(robot.model, q, v * dt)
    # print(q)
    # viz.display(q)
    # time.sleep(0.1)
# a = qpsolvers.solve_ls(r, a, J, h, verbose=True