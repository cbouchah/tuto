import numpy as np

class path_actuator:
    def addNewPathPoint(self, proposedName, aBody, aPosition):
        #porposed name ex: insertion and origin.
        #pyhsical Frame ex: joint name
        #aposition ex: joint position
        proposedName = proposedName
        aBody = aBody
        aPosition = aPosition

    def calcSpeedBetween(self, state_of_the_model_location,state_of_the_model_velocity, path_location, path_velocity):
        location = np.linalg.norm(state_of_the_model_location - path_location)
        velocity = np.linalg.norm(state_of_the_model - aPosition)