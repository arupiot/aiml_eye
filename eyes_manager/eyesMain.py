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
import streamProcessor
import positionFinder
import eyeModel


################################################################################
# Main content of the class.
################################################################################

if __name__ == '__main__':
    # Define detector.
    detector = peopleDetector.peopleDetectorDlib()

    # Define video stream.
    video_stream = streamProcessor.webcamStream()
    
    # Define stream processor based on the video stream and the detector.
#    stream_processor = streamProcessor.streamProcessorFromDetector(video_stream, detector)
    stream_processor = streamProcessor.streamProcessorWithTracker(video_stream, detector, nb_trackers = 5, tracking_time = 100, resize_factor = 2, process_every = 2)

    # Define position finder based on the stream processor.
    # position_finder = positionFinder.videoStreamFinder(video_stream, detector)
#    position_finder = positionFinder.deterministicFinder()
    position_finder = positionFinder.positionFinderFromStreamProcessor(stream_processor)

    # Define eye model based on the position finder.
    eye_model = eyeModel.basicEye(position_finder.getCurrentPosition)

    # Run eye model.
    eye_model.run()

    # Release handle to the webcam (does not work apparently).
    video_stream.close()

