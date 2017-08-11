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

# Package for image processing from the webcam.
import peopleDetector
import scipy.misc
import cv2

# Utilitary packages

################################################################################
# Main content of the class.
################################################################################

class basicEye(ShowBase):
    """
    This class allows for the display of the model.
    """
    def __init__(self, function):
        """
        Initialization of the class.

        :param function: A function with values in [-1, 1] corresponing to the position of detected people.
        """
        # Initialize constructors.
        self.function = function
        # Initialize scene.
        ShowBase.__init__(self)
        # Disable the camera trackball controls.
        self.disableMouse()
        # Load the eye and add texture to it.
        self.scene = self.loader.loadModel("eye_pixar.obj")
        self.myTexture = self.loader.loadTexture("blue-left.png")
        self.scene.setTexture(self.myTexture)
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(1, 1, 1)
        self.scene.setPos(0, 10, 0)
        self.scene.setHpr(60, -90, 0)
        # Place camera.
        self.camera.setPos(0, 0, 0)
        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinEyeTask, "SpinEyeTask")


    def spinEyeTask(self, task):
        """
        This function defines a procedure to move the eye.

        :param task: The current task.
        """
        # Get angle to display. For this model, it should be comprised in [-22, 22].
        angleDegrees = 22 * self.function()
        # Modify eye position.
        self.scene.setHpr(60 - angleDegrees, -90, 0)
        # Return result.
        return Task.cont
