# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:51:45 2017

@author: Lucas
"""

"""
The purpose of this module is to implement a first version of the virtual eyes.
Ultimately, several implementations can be made and compared.
"""

################################################################################
# Imports.
################################################################################

# Packages for mathematical 3D computations.
from math import pi, sin, cos

# Packages for the 3D processing and display.
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3


################################################################################
# Main content of the class.
################################################################################

class basicEye(ShowBase):
    """
    This class allows for the display of the model.
    """
    def __init__(self):
        """
        Initialization of the class.

        :param function: A function with values in [-1, 1] corresponing to the
        position of detected people.
        """
        # Initialize constructors.
        # Initialize scene.
        ShowBase.__init__(self)
        # Disable the camera trackball controls.
        self.disableMouse()
        # Load the eye and add texture to it.
        # self.body = self.loader.loadModel("eyes_rigged_body.egg")
        # self.eyes = self.loader.loadModel("eyes_rigged_eyes.egg")
        self.body = Actor("eyes_rigged_body.egg")
        self.eyes = Actor("eyes_rigged_eyes.egg")
        self.left_eye_joint = self.eyes.controlJoint(None,"modelRoot","eye.L")
        self.joint = self.body.controlJoint(None,"modelRoot","orbicularis03.R")
        print(self.left_eye_joint)
        # print(self.eyes.listJoints())
        # Reparent the model to render.
        self.body.reparentTo(self.render)
        self.eyes.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        # self.scene.setScale(1, 1, 1)
        # self.scene.setPos(0, 0, 0)
        # Place camera.
        self.camera.setHpr(0, 0, 0)
        self.camera.setPos(0, -2, 1.93)
        self.camLens.setFov(5)
        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

    def spinCameraTask(self, task):
        angleDegrees = 10* task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        # self.camera.setPos(2 * sin(angleRadians), -2.0 * cos(angleRadians), 1.9)
        # self.camera.setHpr(angleDegrees, 0, 0)
        self.left_eye_joint.setHpr(0, 30, angleDegrees)
        self.joint.setHpr(angleDegrees, 0, 0)
        return Task.cont


basicEye().run()
