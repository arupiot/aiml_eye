"""
The purpose of this module is to implement functions that return in real time
the locations of detected people in a video stream.

We first implement a class to define a video stream. It must implement the
function

    getCurrentFrame()

which returns the current frame of the stream as np array.

Then we implement classes that analyse such streams. They must implement the
function

    getCurrentLocations()

which returns the locations of detected people in the current frame.
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


class streamProcessorFromDetector:
    """
    This class allows for the processig of a stream using only the given
    detector and applying it to every analysed frame.
    """
    def __init__(self, video_stream, detector, resize_factor = 4.0, process_every = 2):
        """
        Initialization of the class.

        :param video_stream: The video stream being analyzed.
        :param detector: The detector used for the analyse of frames.
        :param resize_factor: Before applying the detector, each frame is resized by this factor. This allows for a faster computation.
        :param process_every: We do not process each frame, but only a fraction of them. We process only one frame in process_every.
        """
        # Initialize constructors.
        self.video_stream = video_stream
        self.detector = detector
        self.resize_factor = resize_factor
        self.process_every = process_every
        # Initialize useful parameters for stream analysis.
        self.frame_counter = 0
        # Initialize the current locations and current image size as [width, height].
        self.current_locations = []
        self.current_image_size = [1, 1]


    def _actualizeLocations(self):
        """
        This function actualizes self.current_locations so that it contains
        the locations corresponding to the current image.
        """
        # Get current frame.
        frame = self.video_stream.getCurrentFrame()
        # Only process a fraction of the frames.
        if (self.frame_counter % self.resize_factor == 0):
            # Resize frame of video for faster face recognition processing
            small_frame = scipy.misc.imresize(frame, 1.0 / self.resize_factor)
            # small_frame = frame
            # Get locations for the normal frame and actualize the current locations.
            self.current_locations = self.resize_factor * np.array(self.detector.getLocations(small_frame))
            # Actualizes the current image size.
            height, width, channels = frame.shape
            self.current_image_size = [width, height]
            # Nullify frame counter to avoid dealing with very large numbers.
            self.frame_counter = 0
        # Increment counter.
        self.frame_counter += 1


    def getCurrentLocations(self):
        """
        This function returns the locations detected in the current image.
        """
        self._actualizeLocations()
        return self.current_locations


    def getCurrentImageSize(self):
        """
        This function returns the current size of the image as [width, height].
        """
        return self.current_image_size


class streamProcessorWithTracker:
    """
    For this stream processor, we use trackers (implemented in the dlib library)
    to improve the speed of the computations.
    """
    def __init__(self, video_stream, detector, nb_trackers = 5, tracking_time = 100, resize_factor = 4, process_every = 2):
        """
        Initialization of the class.

        :param video_stream: The video stream being analyzed.
        :param detector: The detector used for the analyse of frames.
        :param nb_trackers: The maximal number of trackers that we use.
        :param tracking_time: The maximal amount of time a tracker can run.
        :param resize_factor: Before applying the detector, each frame is resized by this factor. This allows for a faster computation.
        :param process_every: We do not process each frame, but only a fraction of them. We process only one frame in process_every.
        """
        # Initialization of constructors.
        self.video_stream = video_stream
        self.detector = detector
        self.nb_trackers = nb_trackers
        self.tracking_time = tracking_time
        self.resize_factor = resize_factor
        self.process_every = process_every
        # Initialize useful parameters for stream analysis.
        self.frame_counter = 0
        # Initialize set of trackers.
        self.trackers = [[dlib.correlation_tracker(), 0, False] for i in range(self.nb_trackers)]
        # Initialize the current locations and current image size as [width, height].
        self.current_locations = []
        self.current_image_size = [1, 1]


    def _non_max_suppression_fast(self, boxes, overlapThresh):
        """
        This function (found on the internet) applies a fast non maxima
        suppression algorith to avoid having multiple locations for only one
        detected object.

        :param boxes: The found locations, as [(left, top, right, bottom)] list.
        :param overlapThresh: The threshold for detecting overlapping boxes.
        """
        # If there are no boxes, return an empty list.
        if len(boxes) == 0:
            return []

        # If the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions.
        if boxes.dtype.kind == "i":
                boxes = boxes.astype("float")

        # Initialize the list of picked indexes.
        pick = []

        # Grab the coordinates of the bounding boxes.
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        # Compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box.
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # Keep looping while some indexes still remain in the indexes
        # list.
        while len(idxs) > 0:
            # Grab the last index in the indexes list and add the
            # index value to the list of picked indexes.
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # Find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box.
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Compute the width and height of the bounding box.
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Compute the ratio of overlap.
            overlap = (w * h) / area[idxs[:last]]

            # Delete all indexes from the index list that have.
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))

        # Return only the bounding boxes that were picked using the
        # integer data type.
        return boxes[pick].astype("int")


    def _actualizeLocations(self):
        """
        This function actualizes self.current_locations so that it contains
        the locations corresponding to the current image.
        """
        # Get current frame.
        frame = self.video_stream.getCurrentFrame()
         # Define array to store all locations.
        all_locations = []
        # Actualise all current trackers.
        for element in self.trackers:
            # Update tracker if available.
            if element[2]:
                element[0].update(frame)
                element[1] += 1
            # Deactivate tracker if it went over tracking time.
            if element[1] >= self.tracking_time:
                element[2] = False

        # We analyse the frame if there is room for trackers.
        if not all([elements[2] for elements in self.trackers]):
            # Resize frame of video for faster face recognition processing.
            small_frame = scipy.misc.imresize(frame, 1.0 / self.resize_factor)
            # Get locations.
            locations = self.resize_factor * np.array(self.detector.getLocations(small_frame))
            # Actualizes the current image size.
            height, width, channels = frame.shape
            self.current_image_size = [width, height]
            # Get available trackers.
            available_trackers_position = [i for i in range(len(self.trackers)) if not self.trackers[i][2]]
            # Fill as much trackers as possible.
            for ([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], i) in zip(locations, available_trackers_position):
                # Get tracker.
                tracker = self.trackers[i][0]
                # Define the dlib rectangle corresponding to the locations.
                rectangle = dlib.rectangle(int(x1), int(y1), int(x3), int(y3))
                # Tracker does not accept empty rectangle.
                if not rectangle.is_empty():
                    tracker.start_track(frame, rectangle)
                    self.trackers[i] = [tracker, 0, True]

        # Get all locations.
        for element in self.trackers:
            tracker = element[0]
            # We add locations of current trackers.
            if element[2]:
                positions = tracker.get_position()
                left, top, right, bottom = positions.left(), positions.top(), positions.right(), positions.bottom()
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                all_locations.append([[left, top], [left, bottom], [right, bottom], [right, top]])
        # Keep only interesting boxes with non maxima suppression.
        boxes = np.array([(left, top, right, bottom) for [[left, top], [left, bottom], [right, bottom], [right, top]] in all_locations])
        boxes = self._non_max_suppression_fast(boxes, 0.5)
        # Actualize locations.
        self.current_locations = [[[left, top], [left, bottom], [right, bottom], [right, top]] for (left, top, right, bottom) in boxes]


    def getCurrentLocations(self):
        """
        This function returns the locations detected in the current image.
        """
        self._actualizeLocations()
        return self.current_locations


    def getCurrentImageSize(self):
        """
        This function returns the current size of the image as [width, height].
        """
        return self.current_image_size
