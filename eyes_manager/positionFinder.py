# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:51:45 2017

@author: Lucas
"""

"""
The purpose of this module is to implement funtions that return in real time
positions of detections.
For each implementation, we want the class to implement a function
    getPosition()
that returns a value in [-1, 1] corresponding to the position of the detection.
"""

################################################################################
# Imports.
################################################################################

# Package for mathematical computation.
import math

# Packages for image processing.
import dlib
import cv2

# Packages for scientific computing.
import scipy.misc
import numpy as np

# Utilitary packages
import time

################################################################################
# Main content of the class.
################################################################################

class deterministicFinder:
    """
    This class implement a finder that periodically returns values in [-1, 1].
    """
    def __init__(self):
        """
        Initialization of the class.
        """
        self.reference_time = time.clock()

    def getPosition(self):
        """
        Returns the wanted position.
        """
        return math.cos(4 * (time.clock() - self.reference_time))

class videoStreamFinder:
    """
    This class returns a finder that analyses the output of a video stream, 
    using a standard detector.
    """
    def __init__(self, video_stream, detector, resize_factor = 3):
        """
        Initialization of the class.

        :param video_stream: The video stream that is analysed in real time.
        :param detector: The detector used to analyse the video_stream.
        """
        # Initialize constructors.
        self.video_stream = video_stream
        self.detector = detector
        # Initialize parameters for video analysis.
        self.resize_factor = resize_factor
        self.process_this_frame = True
        # Initialize wanted position.
        self.currentPosition = 0


    def _positionFromImage(self, image):
        """
        Uses the detector to make the analysis of one image, and returns the
        position of the found detection.

        :param detector: The detector used for the detection of people.
        :param image: The image in which to make the detection.
        :return: A value in [-1, 1] corresponding to the location of the detection
        in the image. -1 correspond to a detection at the very left of the image
        and +1 at the very right. If nothing is detected, returns 0.
        """
        # Get location of detections.
        locations = self.detector.getLocations(image)
        # Get size of image.
        height, width, channels = image.shape
        if len(locations) > 0:
            # Get value for first locations.
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = locations[0]
            value = 2 * ((x1 + x2 + x3 + x4) / (4 * width)) - 1
        else:
            value = 0
        return value


    def _actualizePosition(self):
        """
        Actualizes the current wanted position.
        """
        # Grab a single frame of video
        ret, frame = self.video_stream.read()
        # Resize frame of video for faster face recognition processing
        small_frame = scipy.misc.imresize(frame, 1 / self.resize_factor)
        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find faces and people from the frame.
            self.currentPosition = self._positionFromImage(small_frame)
        # Change frame status
        self.process_this_frame = not self.process_this_frame


    def getPosition(self):
        """
        Returns the wanted position.
        The returned value should be in [-1, 1]
        """
        self._actualizePosition()
        return self.currentPosition
