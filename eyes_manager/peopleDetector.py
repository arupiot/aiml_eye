# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:29:39 2017

@author: Lucas
"""

"""
The goal of this program is to implement different methods to detect
pedestrian in images.
We copy functions from the face_recognition api 
(https://github.com/ageitgey/face_recognition).
"""

###############################################################################
# Imports.
###############################################################################

# Packages used for image processing.
import cv2
import dlib
import numpy as np


###############################################################################
# Definition of global variables.
###############################################################################

RED = (255, 0, 0)
PINK = (255, 0, 255)


###############################################################################
# Main content of the program.
###############################################################################

class peopleDetectorDlib:
    """
    A class for the detection of people using Dlib.
    """
    def __init__(self, detectors = [dlib.fhog_object_detector('dlib_pedestrian_detector.svm'), dlib.get_frontal_face_detector()]):
        """
        Initialization of the class.

        :param detectors: an array of detectors.
        """
        self.detectors = detectors
        self.name = 'Dlib'


    def _trim_css_to_bounds(self, css, image_shape):
        """
        Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :param image_shape: numpy shape of the image array
        :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


    def _rect_to_css(self, rect):
        """
        Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

        :param rect: a dlib 'rect' object
        :return: a plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return rect.top(), rect.right(), rect.bottom(), rect.left()


    def _css_to_locations(self, css):
        """
        Converts a tuple in (top, right, bottom, left) order into array
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        (top, right, bottom, left) = css
        return [[left, top], [left, bottom], [right, bottom], [right, top]]


    def getLocationsByDetector(self, detector, image, number_of_times_to_upsample = 1):
        """
        Returns the locations of the detected objects in the image.

        :param detector: the detector used for the detection.
        :param image: the considered image as numpy array.
        :param number_of_times_to_upsample: used to refine detection but increases
        time of computation.
        """
        locations = detector(image, number_of_times_to_upsample)
        locations = [self._css_to_locations(self._trim_css_to_bounds(self._rect_to_css(people), image.shape)) for people in locations]
        return locations


    def getLocations(self, image, number_of_times_to_upsample = 1):
        """
        Returns all locations for all detectors.
        """
        locations = []
        for detector in self.detectors:
            locations = locations  + self.getLocationsByDetector(detector, image, number_of_times_to_upsample)
        return locations


class peopleDetectorCV:
    """
    A class for the detection of people using the opencv tool.
    """
    def __init__(self):
        """
        Initialization of the class.
        :param hog: the considered hog descriptor for the detection.
        """
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.name = 'CV'


    def _inside(self, r, q):
        """
        Helper function to filter detected locations.

        :param r: rectangle
        :param q: rectangle
        :return: whether r is inside q.
        """
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


    def _rectToLocations(self, css):
        """
        Converts a tuple in (top, right, bottom, left) order into array
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        (top, right, bottom, left) = css
        return [[left, top], [left, bottom], [right, bottom], [right, top]]


    def getLocations(self, image):
        """
        Returns the locations of the detections.
        :param image: the considered image.
        """
        found, w = self.hog.detectMultiScale(image, winStride=(4,4), padding=(16,16), scale=1.05, hitThreshold = 0.25)
        found_filtered = []

        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and self._inside(r, q):
                    break
            else:
                found_filtered.append(r)
        locations = []
        for (x, y, w, h) in found:
            pad_w, pad_h = int(0.15*w), int(0.05*h)
            locations.append(self._rectToLocations((y + pad_h, x + w - pad_w, y +h - pad_h, x + pad_w)))
        return locations


class peopleDetectorBackSub:
    """
    A class for the detection of people using background subtraction with opencv
    tools.
    """
    def __init__(self, fgbg = cv2.createBackgroundSubtractorKNN()):
        """
        Initialization of the class.

        :param fgbg: background subtractor.
        """
        self.fgbg = fgbg
        self.name = 'Background Substraction'


    def getLocations(self, image):
        """
        Get the locations of the detections.

        :param image: the considered image.
        """
        fgmask = self.fgbg.apply(image)
        ret, fgmask = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(fgmask, 10, 20)
        img, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        locations = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            ((x, y), (w, h), angle) = rect
            if w * h > 800:
                box = np.int0(cv2.boxPoints(rect))
                locations.append(box)
        return locations


def drawLocations(image, locations, color = RED):
    """
    Draws the found locations in the image.

    :param image: the considered image.
    :param locations: the locations, as [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] array.
    :param color: the RGB color of the resulting drawing.
    """
    for box in locations:
        cv2.drawContours(image,[np.array(box)],0,(0,0,255),2)
    return image
