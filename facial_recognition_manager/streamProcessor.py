# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 19:05:08 2017

@author: Lucas
"""

"""
The purpose of this module is to implement functions that return in real time
the name, distance, and locations of people in the video stream.
We first implement a class to define a video stream. It must implement the
function
    getCurrentFrame()
which returns the current frame of the stream as np array.
Then we implement classes that analyse such streams. They must implement the
function
    getCurrentAnalysis()
which returns the list [(name, distance, location)] corresponding to the list of
the closest name, the corresponding distance and location, for each detected
location in the image.
"""

################################################################################
# Imports.
################################################################################

# Packages for image processing.
import cv2
import scipy.misc
import numpy as np
import dlib


################################################################################
# Main content of the class.
################################################################################

class webcamStream:
    """
    This class implements the video stream of a webcam.
    """
    def __init__(self, webcam_number = 0):
        """
        Initialization of the class.

        :param webcam_number: The number of the considered webcam (by default 0).
        """
        self.video_capture = cv2.VideoCapture(webcam_number)


    def close(self):
        """
        Closes the process of the stream.
        """
        self.video_capture.release()


    def getCurrentFrame(self):
        """
        Returns the current frame of the stream, as np.array.
        """
        # Grab a single frame of video
        ret, frame = self.video_capture.read()
        return frame


class streamProcessor:
    """
    This class implements the analysis of the stream with the following methods:
        - We analyse only a fraction of the frames.
        - Each analysed frame is internally resized (yet displayed with normal
        size).
        - We average the results over a number of frames.
    """
    def __init__(self, video_stream, face_comparator, nb_frames_to_analyse = 10, resize_factor = 4, process_every = 2):
        """
        Initialization of the class.

        :param video_stream: The video stream being analyzed.
        :param face_comparator: The face comparator used to analyse the frames.
        :param nb_frames_to_analyse: The number of frames over which to average
        the obtained analysis.
        :param resize_factor: Before applying the detector, each frame is
        resized by this factor. This allows for a faster computation.
        :param process_every: We do not process each frame, but only a fraction
        of them. We process only one frame in process_every.
        """
        # Initialization of constructors.
        self.video_stream = video_stream
        self.face_comparator = face_comparator
        self.nb_frames_to_analyse = nb_frames_to_analyse
        self.resize_factor = resize_factor
        self.process_every = process_every
        # Initialization of useful parameters for stream analysis.
        self.frame_counter = 0
        self.frame_history = []
        self.current_frame = None
        # Initialize results.
        self.current_analysis = []
        self.current_name = None


    def _actualizeAnalysis(self, database):
        """
        Actualizes the computation of the analysis.
        :param database: The database with which to compare the frames.
        """
        # Get current frame.
        self.current_frame = self.video_stream.getCurrentFrame()
        # Only process a fraction of the frames.
        if (self.frame_counter % self.process_every == 0):
            # Resize frame of video for faster face recognition processing.
            small_frame = scipy.misc.imresize(self.current_frame, 1 / self.resize_factor)
            self.current_analysis = [(name_match, distance, self.resize_factor * np.array(face_location)) for (name_match, distance, face_location) in self.face_comparator.analyseFrame(small_frame, database)]
            # Actualize frame history.
            if (len(self.frame_history) >= self.nb_frames_to_analyse):
                del self.frame_history[0]
            self.frame_history.append(self.current_analysis)
            # Nullify frame counter to avoid dealing with very large numbers.
            self.frame_counter = 0

        # Analyse frame history.
        frame_history_1 = [element for element in self.frame_history if len(element) == 1]
        results = {}
        for [(name, distance, location)] in frame_history_1:
            results[name] = 0
        for [(name, distance, location)] in frame_history_1:
            results[name] += distance
        results = [(key, results[key]) for key in results]
        results = sorted(results, key = lambda a : -a[1])
        if len(results) == 0:
            self.current_name = None
        elif ((results[0][1] >= 2.5) and (self.current_name == None)):
            self.current_name = results[0][0]
        # Increment counter.
        self.frame_counter += 1


    def getCurrentAnalysis(self):
        """
        Returns the current analysis.
        """
        return self.current_analysis


    def drawCurrentFrame(self, database):
        """
        Draw the current analysis on the current frame and return result as
        np.array.

        :returns: A tuple (clean_frame, drawn_frame) of np.arrays corresponding
        to the current frame.
        """
        self._actualizeAnalysis(database)
        return (self.current_frame, self.face_comparator.drawResult(self.current_frame, self.current_analysis))
