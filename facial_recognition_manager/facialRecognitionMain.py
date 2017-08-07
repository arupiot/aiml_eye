# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:21:04 2017

@author: Lucas
"""

"""
This is the main module for the facial recognition application.
"""

################################################################################
# Imports.
################################################################################

# Packages for the construction of the GUI and analysis of data.
import facialRecognitionGUI
import databaseManager
import facialRecognition
import recommenderSystem
import logFileWriter
import streamProcessor


################################################################################
# Main content of the class.
################################################################################

if __name__ == "__main__":
    # Load database. Modify for empty database.
    database = databaseManager.database()
    #database = database(folder_name = 'Emtpy_database', file_name = 'Empty_database')

    # Load face comparator.
    face_comparator = facialRecognition.faceComparator(tolerance = 0.6)
    # Load recommender system based on the database.
    recommender_system = recommenderSystem.basicRecommender(database)
    # Load log file.
    log_file = logFileWriter.logFile(file_name = 'log.txt', keepLog = True)

    # Initialize video stream.
    video_stream = streamProcessor.webcamStream()
    # Initialize stream processor;
    stream_processor = streamProcessor.streamProcessor(video_stream, face_comparator, nb_frames_to_analyse = 10, resize_factor = 4, process_every = 2)

    # Start the app.
    gui = facialRecognitionGUI.facialRecognitionGui(stream_processor, database,  recommender_system, log_file)
    gui.root.mainloop()
