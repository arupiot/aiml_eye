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
from panda3d.core import Point3, KeyboardButton


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
        #print(dir(ShowBase))
        ShowBase.__init__(self)
        # Disable the camera trackball controls.
        self.disableMouse()
        # Load the eye and add texture to it.
        # self.body = self.loader.loadModel("eyes_rigged_body.egg")
        # self.eyes = self.loader.loadModel("eyes_rigged_eyes.egg")
        self.body = Actor("eyes_rigged_body.egg")
        self.eyes = Actor("eyes_rigged_eyes.egg")
        print(dir(self.body))
        #print(help(self.body.set_light))
        self.left_eye = self.eyes.controlJoint(None,"modelRoot","eye.L")
        self.right_eye = self.eyes.controlJoint(None,"modelRoot","eye.R")
        self.left_orbicularis = self.body.controlJoint(None,"modelRoot","orbicularis03.L")
        self.right_orbicularis = self.body.controlJoint(None,"modelRoot","orbicularis03.R")
        #print(self.left_eye)
        #print(self.left_orbicularis)
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
        self.camLens.setFov(4)
        # Add the spinCameraTask procedure to the task manager.
        #self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        self.taskMgr.add(self.animateEyesTask, "AnimateEyesTask")
        #print(help(self.left_eye_joint.setHpr))
        # self.leftButton = KeyboardButton.asciiKey('d')
        # self.rightButton = KeyboardButton.asciiKey('f')
        self.accept("escape", exit)
        self.accept("a", self.blinkRightEye)
        self.accept("a-up", self.restRightEye)
        self.accept("s", self.blinkLeftEye)
        self.accept("s-up", self.restLeftEye)

    def animateEyesTask(self, task):
        angleRadians = sin(task.time)*.01
        angleDegrees =  angleRadians * 180.0/pi
        #self.left_eye.setHpr(0, 0, angleDegrees)
        #self.left_orbicularis.setHpr(0, 0, 0)
        #print(angleDegrees, angleRadians)
        #self.left_orbicularis.setY()
        return(task.cont)

    def spinCameraTask(self, task):
        angleDegrees = 10* task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(2 * sin(angleRadians), -2.0 * cos(angleRadians), 1.9)
        self.camera.setHpr(angleDegrees, 0, 0)
        return(task.cont)

    def blinkLeftEye(self):
        self.left_orbicularis.setY(.129)

    def restLeftEye(self):
        self.left_orbicularis.setY(.116)

    def blinkRightEye(self):
        self.right_orbicularis.setY(.129)

    def restRightEye(self):
        self.right_orbicularis.setY(.116)


    # def moveTask(self, task):
    #     speed = 0.0
    #     isDown = base.mouseWatcherNode.isButtonDown
    #     if isDown(self.leftButton):
    #         self.left_orbicularis.setY(1)
    #     if isDown(self.rightButton):
    #         self.right_orbicularis.setY(1)
    #     return(task.cont)


basicEye().run()
