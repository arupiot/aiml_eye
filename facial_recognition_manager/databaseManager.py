# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 13:15:27 2017

@author: Lucas
"""

"""
The purpose of this module is to implement the database of faces and information.
We will implement 5 functions :
    - Add(picture, person_name, picture_file_name) to add a person in the database.
    - Remove(link, hardRemove = True) to remove a person from the database
    the hardRemove parameter defines whether we remove physically the information
    from the disk.
    - GetImage(name) to retrieve a picture of a person knowing their name.
    - GetInfo(name) to retrieve the profile information of a person knowing
    their name.
    - GetAllInfo() to retrieve all information of all persons in the database.
"""

###############################################################################
# Imports.
###############################################################################

# Utilitary packages.
import os
import shutil
import re

# Packages used for image processing and numeric computing.
import numpy as np
import facialRecognition
import cv2

# Package used for natural language processing.
import nltk


###############################################################################
# Main content of the program.
###############################################################################

class database:
    """
    A class for the management of the database.
    """
    def __init__(self, folder_name_images = 'Database\\London_images', folder_name_info = 'Database\\London_info', file_name = 'Database\\London_images'):
        """
        Initialization of the class.

        :param folder_name_images: The name of the folder in which images are kept.
        :param file_name: The numpy file containing the encoding information.
        """
        # Initialize useful variables.
        self.file_name = file_name
        self.table_faces = None
        self.folder_name_images = folder_name_images
        self.folder_name_info = folder_name_info

        # Initialize algorithm for facial recognition.
        self.facial_recognition = facialRecognition.faceComparator()

        # Extract the data from the file.
        print('Loading faces from file ' + self.file_name)
        self.table_faces = np.load(self.file_name + '.npy')
        print('Loaded ' + str(len(self.table_faces)) + ' faces.')


    def add(self, frame, face_name, file_name, check_name = True):
        """
        Attempts to update the database adding picture in path 'training_images\\face_name\\file_name.jpg'.
        It computes and returns 2 booleans : name_already_exists and one_face_detected,
        and the database is updated iff (one_face_detected && !name_already_exists).

        :param frame: The image.
        :param face_name: The name of the person in the image.
        :param file_name: The filename we want to give to the image.
        :param check_name: Boolean to decide whether we check if the name is
        already in the database.
        :return: Two booleans, name_already_exists and one_face_detected, whose
        value show whether the database was successfully updated or not, and
        the correponding problem if not.
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
                os.mkdir(self.folder_name_images + '\\' + face_name)
            except Exception:
                name_already_exists = True

        if one_face_detected and not name_already_exists:
            # Save data
            link = self.folder_name_images + '\\' + face_name + '\\' + file_name + '.jpg'
            self.table_faces = np.append(self.table_faces, [(encodings[0], face_name, link)], axis = 0)
            cv2.imwrite(self.folder_name_images + '\\' + face_name + '\\' + file_name + '.jpg', frame)
            np.save(self.file_name, self.table_faces)
        # Return results.
        return name_already_exists, one_face_detected


    def remove(self, link, hard_remove = True):
        """
        Remove the corresponding link image from the database.
        If hard_remove, we also delete physically the image from the computer.

        :param link: The link to the image we want to erase.
        :param hard_remove: Parameter to decide wheter or not we physically
        erase the image from the computer.
        """
        # Remove the corresponding list to table_faces.
        self.table_faces = list(filter(lambda x : x[2] != link, self.table_faces))
        # Try to remove from folder if required.
        if hard_remove:
            try:
                os.remove(link)
                # We even delete the folder if it is empty.
                link_to_folder = link.split('\\')[0] + '\\' +link.split('\\')[1]
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
        :return: The corresponding image.
        """
        # Get to folder.
        link_folder = self.folder_name_images + '\\' + name
        list_images = os.listdir(link_folder)
        return cv2.imread(link_folder + '\\' + list_images[0])


    def getInfo(self, name):
        """
        Obtain the information corresponding to the name.

        :param name: The considered name.
        :return: A list of words extracted from the description of the
        corresponding person.
        """
        # Get to location.
        filename = self.folder_name_info + '\\' + name + '\\' + name + '_.txt'
        # Read content of file.
        with open(filename, 'r') as file:
            # Obtain card and bio.
            file_content = file.read()
            file_card = file_content.split('CARD:\n')[1].split('BIO:\n')[0]
            file_bio = file_content.split('BIO:\n')[1]
        # From card, extract name, job, office.
        people_name = file_card.split('\n')[0]
        people_job = file_card.split('\n')[1]
        people_office = file_card.split('\n')[2]
        # From bio, extract different categories and content for each category.
        bio_info = [(content.split('\n')[0], content.split('\n', 1)[-1]) for content in file_bio.split('\n\n')]
        # Define interesting words.
        interesting_words_individual = []
        # Add job description.
        if len(people_job.split(',')) == 2:
            for word in people_job.casefold().split(',')[0].split(' '):
                interesting_words_individual.append(word.casefold())
        # Add content
        for (categories, content) in bio_info:
            for word in re.split('\W+', content):
                interesting_words_individual.append(word.casefold())
        # Remove useless words.
        interesting_words_individual = list(filter(lambda a: a != '', interesting_words_individual))
        return interesting_words_individual


    def getAllInfo(self):
        """
        Obtain all the information of all persons in the database. We filter
        it using a stopwords list.

        :return: A dictionary correspondig to the frequency distribution of the
        words in all profiles.
        """
        # Define stopwords for the filtering of information
        stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'also']

        # Extract info from all files.
        people_info = []
        # Browse all names.
        for name in os.listdir(self.folder_name_info):
            filename = self.folder_name_info + '\\' + name + '\\' + name + '_.txt'
            # Read content of file.
            with open(filename, 'r') as file:
                # Obtain card and bio.
                file_content = file.read()
                file_card = file_content.split('CARD:\n')[1].split('BIO:\n')[0]
                file_bio = file_content.split('BIO:\n')[1]
                # From card, extract name, job, office.
                people_name = file_card.split('\n')[0]
                people_job = file_card.split('\n')[1]
                people_office = file_card.split('\n')[2]
                # From bio, extract different categories and content for each category.
                bio_info = [(content.split('\n')[0], content.split('\n', 1)[-1]) for content in file_bio.split('\n\n')]
                # Update info file.
                people_info.append((people_name, people_job, people_office, bio_info))

        # Get list of interesting words for each person.
        interesting_words = []
        for (people_name, people_job, people_office, bio_info) in people_info:
            interesting_words_individual = []
            # Add job description.
            if len(people_job.split(',')) == 2:
                for word in people_job.casefold().split(',')[0].split(' '):
                    interesting_words_individual.append(word.casefold())
            # Add content
            for (categories, content) in bio_info:
                for word in re.split('\W+', content):
                    interesting_words_individual.append(word.casefold())
            # Remove useless words.
            interesting_words_individual = list(filter(lambda a: a != '', interesting_words_individual))
            # Append result
            for word in interesting_words_individual:
                interesting_words.append(word)
        # Compute freuency distribution.
        fdist = nltk.FreqDist([word for word in interesting_words if not word in stopwords])

        # Return result.
        return fdist
