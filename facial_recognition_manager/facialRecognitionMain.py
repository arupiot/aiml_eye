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

# Package for video capture.
import cv2

# Utilitary packages.
import time

################################################################################
# Main content of the class.
################################################################################

if __name__ == "__main__":
    # Load database. Modify for empty database.
    database = databaseManager.database()
    #database = database(folder_name = 'Emtpy_database', file_name = 'Empty_database')

    # Load face comparator.
    facialRecognition = facialRecognition.faceComparator(tolerance = 0.6)
    # Load recommender system.
    recommenderSystem = recommenderSystem.basicRecommender(database)
    # Load log file.
    log_file = logFileWriter.logFile(file_name = 'log.txt', keepLog = True)

    # Initialize the video stream and allow the camera sensor to warmup.
    print("Warming up camera...")
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    # Start the app.
    gui = facialRecognitionGUI.facialRecognitionGui(vs, database, facialRecognition, recommenderSystem, log_file)
    gui.root.mainloop()
