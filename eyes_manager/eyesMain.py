# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:46:44 2017

@author: Lucas
"""

"""
This is the main module for the eyes application.
"""

################################################################################
# Imports.
################################################################################

# Packages.
import peopleDetector
import positionFinder
import eyeModel

# Package for webcam capture.
import cv2

################################################################################
# Main content of the class.
################################################################################

if __name__ == '__main__':
    # Define detector.
    detector = peopleDetector.peopleDetectorDlib()

    # Define video stream.
    video_stream = cv2.VideoCapture(0)

    # Define position finder.
    # position_finder = positionFinder.videoStreamFinder(video_stream, detector)
    position_finder = positionFinder.deterministicFinder()

    # Define eye model.
    eye_model = eyeModel.basicEye(position_finder.getPosition)

    # Run eye model.
    eye_model.run()

    # Release handle to the webcam (does not work apparently).
    video_capture.release()
