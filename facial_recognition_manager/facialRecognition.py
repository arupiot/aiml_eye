# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:57:35 2017

@author: Lucas
"""

"""
The purpose of this module is to implement the algorithm of face recognition.
For that, we use the Dlib library.
We also copy functions from the face_recognition api
(https://github.com/ageitgey/face_recognition).
"""

###############################################################################
# Imports.
###############################################################################

# Packages for image processing and numeric computations.
import scipy.misc
import dlib
import numpy as np
import cv2


###############################################################################
# Definition of global variables.
###############################################################################

RED = (255, 0, 0)
PINK = (255, 0, 255)
WHITE = (255, 255, 255)


###############################################################################
# Main content of the module.
###############################################################################

class faceComparator:
    """
    A class to implement useful functions regarding the detection and drawing of
    pictures.
    """
    def __init__(self, tolerance = 0.55):
        """
        Initialization of the class.

        :param folder_name_images: the name of the folder in which images are kept.
        :param file_name: the numpy file containing the encoding information.
        """
        # Initialize useful variables.
        self.tolerance = tolerance

        # Initialize models.
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor_model = 'Models\\shape_predictor_68_face_landmarks.dat'
        self.pose_predictor = dlib.shape_predictor(self.predictor_model)
        self.face_recognition_model = 'Models\\dlib_face_recognition_resnet_model_v1.dat'
        self.face_encoder = dlib.face_recognition_model_v1(self.face_recognition_model)


    def _rect_to_css(self, rect):
        """
        Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

        :param rect: a dlib 'rect' object
        :return: a plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return rect.top(), rect.right(), rect.bottom(), rect.left()


    def _css_to_rect(self, css):
        """
        Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :return: a dlib `rect` object
        """
        return dlib.rectangle(css[3], css[0], css[1], css[2])


    def _trim_css_to_bounds(self, css, image_shape):
        """
        Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :param image_shape: numpy shape of the image array
        :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


    def face_distance(self, face_encodings, face_to_compare):
        """
        Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.

        :param faces: List of face encodings to compare
        :param face_to_compare: A face encoding to compare against
        :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
        """
        if len(face_encodings) == 0:
            return np.empty((0))

        return np.linalg.norm(face_encodings - face_to_compare, axis=1)


    def load_image_file(self, filename, mode='RGB'):
        """
        Loads an image file (.jpg, .png, etc) into a numpy array

        :param filename: image file to load
        :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
        :return: image contents as numpy array
        """
        return scipy.misc.imread(filename, mode=mode)


    def _raw_face_locations(self, img, number_of_times_to_upsample=1):
        """
        Returns an array of bounding boxes of human faces in a image

        :param img: An image (as a numpy array)
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :return: A list of dlib 'rect' objects of found face locations
        """
        return self.face_detector(img, number_of_times_to_upsample)


    def face_locations(self, img, number_of_times_to_upsample=1):
        """
        Returns an array of bounding boxes of human faces in a image

        :param img: An image (as a numpy array)
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :return: A list of tuples of found face locations in css (top, right, bottom, left) order
        """
        return [self._trim_css_to_bounds(self._rect_to_css(face), img.shape) for face in self._raw_face_locations(img, number_of_times_to_upsample)]


    def _raw_face_landmarks(self, face_image, face_locations=None):
        if face_locations is None:
            face_locations = self._raw_face_locations(face_image)
        else:
            face_locations = [self._css_to_rect(face_location) for face_location in face_locations]

        return [self.pose_predictor(face_image, face_location) for face_location in face_locations]


    def face_landmarks(self, face_image, face_locations=None):
        """
        Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image

        :param face_image: image to search
        :param face_locations: Optionally provide a list of face locations to check.
        :return: A list of dicts of face feature locations (eyes, nose, etc)
        """
        landmarks = self._raw_face_landmarks(face_image, face_locations)
        landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

        # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
        return [{
            "chin": points[0:17],
            "left_eyebrow": points[17:22],
            "right_eyebrow": points[22:27],
            "nose_bridge": points[27:31],
            "nose_tip": points[31:36],
            "left_eye": points[36:42],
            "right_eye": points[42:48],
            "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
            "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
        } for points in landmarks_as_tuples]


    def face_encodings(self, face_image, known_face_locations=None, num_jitters=1):
        """
        Given an image, return the 128-dimension face encoding for each face in the image.

        :param face_image: The image that contains one or more faces
        :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
        :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
        :return: A list of 128-dimentional face encodings (one for each face in the image)
        """
        raw_landmarks = self._raw_face_landmarks(face_image, known_face_locations)

        return [np.array(self.face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


    def compare_faces(self, known_face_encodings, face_encoding_to_check, tolerance=0.6):
        """
        Compare a list of face encodings against a candidate encoding to see if they match.

        :param known_face_encodings: A list of known face encodings
        :param face_encoding_to_check: A single face encoding to compare against the list
        :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
        :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
        """
        return list(self.face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


    def computeDistances(self, face_encoding, database):
        """
        Compute the list of distances and names relatively to the encodings of
        all the faces in the database.

        :param face_encoding: The encoding to compare.
        :param database: The database to search.
        """
        # TODO: improve the code.
        return [(self.face_distance([known_encoding], face_encoding)[0], name) for (known_encoding, name, link) in database.table_faces]


    def analyseFrame(self, frame, database):
        """
        Returns the name of the person corresponding to closest face in the database,
        along with the corresponding distances, and the corresponding locations.

        :param frame: The image to analyse.
        :param database: The database to search.
        :return: A list [(name, distance, location)] corresponding to the identified names
        and distances in the image.
        """
        # Find all the faces and face encodings in the current frame of video.
        face_locations = self.face_locations(frame)
        face_encodings = self.face_encodings(frame, face_locations)
        result = []
        for i in range(len(face_locations)):
            face_location = face_locations[i]
            face_encoding = face_encodings[i]
            # See if the face is a match for the known face(s).
            distances = self.computeDistances(face_encoding, database)
            name_match = "Unknown"
            # If database is empty, we impose distance = 1.
            if len(distances) == 0:
                distance = 1
            # Else, we compute closest distance and name.
            else:
                (distance, name) = min(distances)
            # If the distance is smaller than tolerance, we keep the found name.
            if distance <= self.tolerance:
                name_match = name
            # Compute result array.
            result.append((name_match, distance, face_location))
        return result


    def drawResult(self, frame, result, color_box = RED, color_text = WHITE):
        """
        Display the obtained results.

        :param frame: The image to modify and display.
        :param result: A list [(name, distance, location)] corresponding to the identified names
        and distances in the image.
        :param color: The color we draw.
        :return: A modified frame: we draw a bow around the face, and a label
        with the name and distance below the face.
        """

        # Display the results.
        for (name, distance, ((top, right, bottom, left))) in result:
            # Draw a box around the face.
            cv2.rectangle(frame, (left, top), (right, bottom), color_box, 2)
            # Draw a label below the face.
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color_box, cv2.FILLED)
            # Draw name and distance.
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 20), font, 0.5, color_text, 1)
            cv2.putText(frame, str(1 - distance), (left + 6, bottom - 5), font, 0.5, color_text, 1)
        return frame


    def findSimilarFaces(self, frame, database, nb_faces = 3):
        """
        Returns the closest matches of the person in the frame relatively to the database.
        The function requires that there is one and only one person in the database.
        If this is not the case, it returns an exception "Wrong number of faces".
        The function should return different names, even though it is possible
        that some people have more than one picture of their face in the database.

        :param frame: The image to analyse.
        :param database: The database to search for matches.
        :param nb_faces: The number of results retrieved from the database.
        :return: A list [(distance, name)] with nb_face elements, sorted by distance.
        """
        # Find all the faces and face encodings in the current frame of video.
        face_locations = self.face_locations(frame)
        face_encodings = self.face_encodings(frame, face_locations)

        # We proceed if we notice only one face.
        if len(face_encodings) == 1:
            for face_encoding in face_encodings:
                # Compare with known faces.
                distances = self.computeDistances(face_encoding, database)
                # Sort the array of distances.
                similar_faces = sorted(distances)
                # Get the unique name values.
                u, indices = np.unique([name for (distance, name) in similar_faces], return_index=True)
                closest_unique_faces_indices = indices[np.argpartition(indices, (1, nb_faces))[:nb_faces]]
                similar_faces = [similar_faces[i] for i in closest_unique_faces_indices]
                # Extract values.-
                return similar_faces
        else:
            raise Exception('Wrong number of faces.')
