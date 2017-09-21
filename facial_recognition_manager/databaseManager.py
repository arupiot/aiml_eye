"""
The purpose of this module is to implement the database of faces and information.

We will implement 4 functions :

    - Add(picture, person_name, picture_file_name) to add a person in the database.
    - Remove(link, hardRemove = True) to remove a person from the database. the hardRemove parameter defines whether we remove physically the information from the disk.
    - GetImage(name) to retrieve a picture of a person knowing their name.
    - GetProfile(name) to retrieve the profile of a person knowing their name.

Our implementation is based on an array of the form

    [(encodings, name, path_to_image, profile)]

for every stored image.
"""

###############################################################################
# Imports.
###############################################################################

# Utilitary packages.
import os
import shutil
import re
from os.path import join, split
import os.path

# Packages used for image processing and numeric computing.
import numpy as np
import facialRecognition
import cv2


###############################################################################
# Main content of the program.
###############################################################################

class database:
    """
    A class for the management of the database.
    """
    def __init__(self, file_name = join('Database','London_database')):
        """
        Initialization of the class.

        :param file_name: Filename to the numpy file containing the encoding
        information. It must be an array of the form [(encodings, name, path_to_image, profile)]
        """
        # Initialize constructor.
        self.file_name = file_name

        # Open file.
        print('Loading faces from file ' + self.file_name)
        self.table_faces = np.load(self.file_name + '.npy')
        print('Loaded ' + str(len(self.table_faces)) + ' faces.')

        # Get path to images folder.
        self.folder_name_images = join(split(self.file_name)[0], split(split(self.table_faces[0][2])[0])[0])

        # Initialize algorithm for facial recognition.
        self.facial_recognition = facialRecognition.faceComparator()

        # Default profile value.
        self.DEFAULT_PROFILE = 'No Arup People profile'


    def add(self, frame, face_name, file_name, check_name = True):
        """
        Attempts to update the database adding picture in path 'self.folder_name_images/face_name/file_name.jpg'.
        It computes and returns 2 booleans : name_already_exists and one_face_detected,
        and the database is updated iff (one_face_detected && !name_already_exists).

        :param frame: The image.
        :param face_name: The name of the person in the image.
        :param file_name: The filename we want to give to the image.
        :param check_name: Boolean to decide whether we check if the name is already in the database.
        :return: Two booleans, name_already_exists and one_face_detected, whose value show whether the database was successfully updated or not, and the correponding problem if not.
        """
        # Initialise validity booleans.
        name_already_exists = False
        one_face_detected = True

        # Compute encodings, and input validity.
        encodings = self.facial_recognition.face_encodings(frame)
        if len(encodings) != 1:
            one_face_detected = False

        if check_name and one_face_detected:
            try:
                # Create dedicated folder and add picture.
                os.mkdir(join(self.folder_name_images, face_name))
            except Exception:
                name_already_exists = True

        if one_face_detected and not name_already_exists:
            # Save data
            link = join(self.folder_name_images, face_name, file_name) + '.jpg'
            self.table_faces = np.append(self.table_faces, [(encodings[0], face_name, link, self.DEFAULT_PROFILE)], axis = 0)
            cv2.imwrite(join(self.folder_name_images, face_name, file_name) + '.jpg', frame)
            np.save(self.file_name, self.table_faces)
        # Return results.
        return name_already_exists, one_face_detected


    def remove(self, link, hard_remove = True):
        """
        Remove the corresponding link image from the database.
        If hard_remove, we also delete physically the image from the computer.

        :param link: The link to the image we want to erase.
        :param hard_remove: Parameter to decide whether or not we physically erase the image from the computer.
        """
        # Remove the corresponding list to table_faces.
        self.table_faces = list(filter(lambda x : x[2] != link, self.table_faces))
        # Try to remove from folder if required.
        if hard_remove:
            try:
                os.remove(link)
                # We even delete the folder if it is empty.
                link_to_folder = os.path.split(link)[0]
                if len(os.listdir(link_to_folder)) == 0:
                    shutil.rmtree(link_to_folder)
            except Exception:
                print('Hard remove did not succeed.')
        # Actualize database.
        np.save(self.file_name, self.table_faces)


    def getImage(self, name):
        """
        Returns the image corresponding to the name.

        :param name: The considered name.
        :return: The corresponding image. Returns None if no image is found (even though it should not happen).
        """
        try:
            # Get link to image from the table of encodings.
            link = [path_to_image for (encoding, user_name, path_to_image, profile) in self.table_faces if user_name == name][0]
            # Return cv2 image.
            return cv2.imread(link)
        except:
            # Return default value.
            return None


    def getProfile(self, name):
        """
        Returns the profile corresponding to the name.

        :param name: The considered name.
        :return: A string corresponding to the profile. Either precomputed profile or 'Unable to match description with profile' if name is not in the database (even though it should not happen).
        """
        try:
            # Get profile from the table of encodings.
            return [profile for (encoding, user_name, path_to_image, profile) in self.table_faces if user_name == name][0]
        except:
            # Return default value.
            return self.DEFAULT_PROFILE
