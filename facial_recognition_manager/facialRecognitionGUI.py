# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:21:04 2017

@author: Lucas
"""

"""
The purpose of this module is to implement tools for the Graphical User Interface
associated with the facial recognition algorithm.
"""

###############################################################################
# Imports.
###############################################################################

# Packages for graphical user interface.
import tkinter as tk
import tkinter.simpledialog as simpledialog
import tkinter.messagebox as messagebox
import threading

# Packages for image processing.
import cv2
from PIL import Image
from PIL import ImageTk
import scipy
import numpy as np


###############################################################################
# Main content of the module.
###############################################################################


class facialRecognitionGui:
    """
    A class for the layout of the facial recognition Graphical User Interface.
    """
    # Initialize the class.
    def __init__(self, stream_processor, database,  recommenderSystem, logFile):
        """
        Initialisation of the class.

        :param database: The database storing images, encodings and info of people.
        :param faceRecognition: The facial recognition tool.
        :param logFile: The logFileWriter object we will use to store information.
        """
        # Store the video stream object and output path, then initialize
        # most recently read frame, thread for reading frames, and
        # thread stop event.
        self.stream_processor = stream_processor
        self.frame = None
        self.cleanFrame = None
        self.thread = None
        self.stopEvent = None

        # Initialize useful variables for data analysis.
        self.counter_frame = 0
        self.process_every = 2
        self.resize_factor = 4

        # Initialize database, face comparator, and log file.
        self.database = database
        self.log_file = logFile
        self.recommenderSystem = recommenderSystem

        # Initialize the root window and image panel (uncomment for fullscreen).
        self.root = tk.Tk()
        # self.root.attributes('-fullscreen',True)
        self.panel = None

        # Background.
        self.background = ImageTk.PhotoImage(Image.open('background.jpg'))
        background_label = tk.Label(self.root, image=self.background)
        # background_label.place(x=0, y=0, relwidth=1, relheight=1)
        background_label.place(x = 0, y = 0)

        # Create the main sections of the layout,
        # and lay them out.
        top = tk.Frame(self.root)
        bottom = tk.Frame(self.root)
        top.pack(side=tk.TOP)
        bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

        # Create the widgets for the top part of the GUI,
        # and lay them out.
        save_face_button = tk.Button(self.root, text="Capture face", width=20, height=2, command=self.captureFace)
        check_similar_faces_button = tk.Button(self.root, text="Check for similar faces", width=20, height=2, command=self.checkSimilarFaces)
        save_face_button.pack(in_=bottom, side=tk.LEFT, fill="none", expand=True)
        check_similar_faces_button.pack(in_=bottom, side=tk.LEFT, fill="none", expand=True)

        # Start a thread that constantly pools the video sensor for
        # the most recently read frame.
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # Set a callback to handle when the window is closed.
        self.root.wm_title("Face Detection")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)


    def _processFrameForDisplay(self, frame):
        """
        Converts frame from cv image to tk image, in order to display it.

        :param frame: The image to convert.
        :return: The converted image.
        """
        b,g,r = cv2.split(frame)
        frame = cv2.merge((r,g,b))
        image = Image.fromarray(frame)
        #print(dir(image))
        #image.show()
        image = ImageTk.PhotoImage(image)
        return image


    def videoLoop(self):
        """
        Manage the video loop.
        """
        try:
            while not self.stopEvent.is_set():
                # Get frames.
                print('processing frame')
                (self.cleanFrame, self.frame) = self.stream_processor.drawCurrentFrame(self.database)

                # Process the current frame for display.
                image = self._processFrameForDisplay(self.frame)

                # If the panel is not None, we need to initialize it.
                if self.panel is None:
                    self.panel = tk.Label(self.root, image=image)
                    self.panel.image = image
                    self.panel.pack(side="right", padx=10, pady=10)

                # Otherwise, simply update the panel.
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

        except RuntimeError:
            print("Caught a RuntimeError.")


    def captureFace(self):
        """
        Action for the 'capture face button'.
        """
        # Keep log of action.
        self.log_file.message('Action: capture face.')
        try:
            # Save the current frame.
            current_frame = self.cleanFrame
            # Open dialog to obtain the name.
            name = simpledialog.askstring('Enter your name', 'Name')
            # Attempt to add picture to the database.
            name_already_exists, one_face_detected = self.database.add(current_frame, name, name + '_0', check_name = True)
            # Handle exceptions.
            if name_already_exists:
                self.log_file.message('Action unsuccessful: name ' + name + ' already exists.')
                messagebox.showerror('Error', 'This name already exists.')
            elif not one_face_detected:
                self.log_file.message('Action unsuccessful: one and only one face has to be detected.')
                messagebox.showerror('Error', 'An error has occured.\nPlease make sure that one and only one face is detected by the camera.')
            else:
                # Action successful, keep log.
                self.log_file.message('Action successful: ' + name + ' added to the database.')
                # Add more pictures.
                nb_pictures = 1
                while(messagebox.askyesno(title = name, message = 'Take another picture of you, ' + name + '?')):
                    # Save the current frame.
                    current_frame = self.cleanFrame
                    # Attempt to add picture to the database.
                    name_already_exists, one_face_detected = self.database.add(current_frame, name, name + '_' + str(nb_pictures), check_name = False)
                    # Handle exceptions.
                    if not one_face_detected:
                        self.log_file.message('Action unsuccessful: one and only one face has to be detected.')
                        messagebox.showerror('Error', 'An error has occured.\nPlease make sure that one and only one face is detected by the camera.')
                    else:
                        self.log_file.message('Action successful: took another picture for ' + name + '.')
                        nb_pictures += 1

        except Exception:
            pass


    def checkSimilarFaces(self):
        """
        Actions for the 'check similar faces' button.
        We check for the 3 most similar faces in the database.
        """
        # Keep log of action.
        self.log_file.message('Action: check for similar faces.')
        # Save the current frame.
        current_frame = self.cleanFrame
        # Compute names corresponding to similar faces.
        try:
            similar_faces = self.faceRecognition.findSimilarFaces(current_frame, self.database, nb_faces = 3)
            # Process the self frame for display.
            width = current_frame.shape[1]
            current_frame = cv2.resize(current_frame, (0, 0), fx = 200 / width, fy = 200 / width)
            processed_current_frame = self._processFrameForDisplay(current_frame)

            # Obtain and process other frames for display.
            images_faces = []
            for (i, (distance, name)) in zip(range(len(similar_faces)), similar_faces):
                image = self.database.getImage(name)
                width = image.shape[1]
                resized_image = cv2.resize(image, (0, 0), fx = 200 / width, fy = 200 / width)
                images_faces.append((resized_image, name, distance))

            # Compute useful variables for display.
            max_height = max(current_frame.shape[0], max([image.shape[0] for (image, name, distance) in images_faces]))
            total_length = 210 * (len(images_faces) + 1) + 30

            # Open new window.
            window = tk.Toplevel(self.root)
            window.geometry("%dx%d%+d%+d" % (total_length, max_height, 250, 125))

            # Display current frame in the window.
            lbl_image = tk.Label(window, image = processed_current_frame)
            lbl_image.image = processed_current_frame
            lbl_image.place(x = 10, y = max_height - current_frame.shape[0])
            lbl_text = tk.Label(window, text = 'You')
            lbl_text.text = 'You'
            lbl_text.place(x = 10, y = max_height - current_frame.shape[0])

            # Display other similar faces.
            for (i, (image, name, distance)) in zip(range(len(images_faces)), images_faces):
                processed_image = self._processFrameForDisplay(image)
                lbl_image = tk.Label(window, image = processed_image)
                lbl_image.image = processed_image
                lbl_image.place(x = 210 * (i + 1) + 30, y = max_height - image.shape[0])
                lbl_text = tk.Label(window, text = name + '\nProximity: ' + str(round(1 - distance, 3)))
                lbl_text.text = name + '\nProximity: ' + str(round(1 - distance, 3))
                lbl_text.place(x = 210 * (i + 1) + 30, y = max_height - image.shape[0])
        except Exception as e:
            # Open new window and add box error.
            messagebox.showerror('Error', 'An error has occured.\nPlease make sure that one and only one face is detected by the camera.')


    def onClose(self):
        """
        Close the window and all events.
        """
    		# Set the stop event, cleanup the camera, and allow the rest of
    		# the quit process to continue.
        self.stopEvent.set()
        self.root.destroy()
        self.stream_processor.video_stream.close()
        print("Closing...")
