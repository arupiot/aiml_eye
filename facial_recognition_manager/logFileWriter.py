# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:22:45 2017

@author: Lucas
"""

"""
The purpose of this module is to implement a class allowing us to keep logs
of the different actions realised.
The analysis of such files may help to interpret the reception of the exhibit.
"""


###############################################################################
# Imports.
###############################################################################

# Utilitary packages.
import time


###############################################################################
# Main content of the module.
###############################################################################

class logFile:
    """
    A class to keep logs of the different actions.
    """
    def __init__(self, file_name = 'log.txt', keepLog = True):
        """
        Initialization of the class.
        Note that if keepLog is False, we don't do anything.

        :param file_name: The name of the file in which we keep log.
        :param keepLog: Parameter to define whether we actually store the information or not.
        """
        # Initialize variables.
        self.file_name = file_name
        self.keepLog = keepLog
        # We only keep log if asked.
        if self.keepLog:
            # Append date.
            with open(self.file_name, 'a') as file:
                file.write('Application opened on ' + time.strftime("%c") + '\n')


    def message(self, string):
        """
        Append message to the file.

        :param string: The considered message to log.
        """
        # We only append message if asked.
        if self.keepLog:
            with open(self.file_name, 'a') as file:
                tag = '[' + time.strftime("%c") + ']: '
                file.write(tag + string + '\n')
