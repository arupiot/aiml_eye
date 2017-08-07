# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:51:45 2017

@author: Lucas
"""

"""
The purpose of this module is to implement funtions that return in real time
positions of detections.
For each implementation, we want the class to implement a function
    getCurrentPosition()
that returns a value in [-1, 1] corresponding to the current position of the 
detection.
"""

################################################################################
# Imports.
################################################################################

# Package for mathematical computation.
import math

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

    def getCurrentPosition(self):
        """
        Returns the wanted position.
        """
        return math.cos(4 * (time.clock() - self.reference_time))


class positionFinderFromStreamProcessor:
    """
    This class implements a finder that uses in real time the results of the
    stream processor.
    """
    # TODO: Implement an averaging algorithm for smoother returns.
    def __init__(self, stream_processor):
        """
        Initialization of the class.
        
        :param stream_processor: The stream processor the class relies on.
        """
        # Initialize constructor for the class.
        self.stream_processor = stream_processor
        # Initialize current position.
        self.currentPosition = 0
    
    def getCurrentPosition(self):
        """
        Returns the current position of the found detection, based on the 
        results of the stream processor.
        
        :return: A value in [-1, 1] corresponding to the location of the detection
        in the image. -1 correspond to a detection at the very left of the image
        and +1 at the very right. If nothing is detected, returns 0.
        """
        # Get current locations of detections.
        current_locations = self.stream_processor.getCurrentLocations()
        # Get current image size.
        [width, height] = self.stream_processor.getCurrentImageSize()
        # Compute the output value. We take the barycenter of the first 
        # location and normalizes it by the width of the image.
        if len(current_locations) > 0:
            # Get value for first locations.
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = current_locations[0]
            value = 2 * ((x1 + x2 + x3 + x4) / (4 * width)) - 1
        else:
            value = 0
        return value