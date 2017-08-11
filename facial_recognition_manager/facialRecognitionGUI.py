"""
The purpose of this module is to implement the graphical user interface for
the face recognition application.

For that, we use the kivy library.
"""

################################################################################
# Imports.
################################################################################

# Imports for the graphical user interface.
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput

from kivy.properties import StringProperty

from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# Start application with full screen.
# from kivy.core.window import Window
# Window.fullscreen = 'auto'

# Imports for Image analysis.
import cv2

# Modules for the analysis of data.
import databaseManager
import facialRecognition
import logFileWriter
import streamProcessor

# Utilitary.
import time

################################################################################
# Definition of the graphical user interface.
################################################################################


# Define and build the layout. We insert elements into box layouts.
class FaceRecognitionGui(BoxLayout):
    """
    This class defines the global layout for the graphical user interface.
    """
    pass


class OutputLayout(BoxLayout):
    """
    This class defines the layout for the output frame.
    """
    pass

# Load the layout from string (utlimately from .kv file).
Builder.load_string('''
<OutputLayout>
    canvas.before:
        Color:
            rgba: 156, 25, 26, 0.5
        Rectangle:
            pos: self.pos[0], self.pos[1] + 100
            size: self.size[0], self.size[1] * 0.8

    BoxLayout:
        id: output
        orientation: 'vertical'

        Label:
            id: output_hello
            markup: True
            text: '[color=000000][b]Hello[/b][/color]'
            font_size: 65
            size_hint: 1, .5
            pos_hint: {'center_x':.5, 'center_y': .5}

        Label:
            id: output_text
            markup: True
            text: app.output_text
            font_size: 35
            size_hint: 1, .5
            pos_hint: {'center_x':.5, 'center_y': .5}


<FaceRecognitionGui>
    orientation: 'vertical'

    canvas.before:
        BorderImage:
            border: 0, 0, 0, 0
            source: 'background.jpg'
            pos: self.pos
            size: self.size

    BoxLayout:
        Image:
            id: webcam
            size_hint: 1.5, 1.5
            pos_hint: {'center_x':.5, 'center_y': .5}

        OutputLayout:
            id: output_layout

    Button:
        text: 'Capture face'
        size_hint: 0.4, 0.2
        pos_hint: {'center_x':.5, 'center_y': .5}
        on_press: app._buttonCommandCaptureFace()
''')


class MainApp(App):
    """
    Main class, corresponding to the application.
    """
    # Title of the app.
    title = 'Face Recognition'
    # The output text.
    output_text = StringProperty('')

    def build(self):
        """
        Builder for the class.
        We initialize the stream processor.
        We then schedule the clock.
        We return the layout.
        """
        # Load database. Modify for empty database.
        self.database = databaseManager.database()
        # Load face comparator.
        self.face_comparator = facialRecognition.faceComparator(tolerance = 0.6)
        # Load log file. TODO: log different actions.
        self.log_file = logFileWriter.logFile(file_name = 'log.txt', keepLog = False)
        # Initialize video stream.
        self.video_stream = streamProcessor.webcamStream()
        # Initialize stream processor.
        self.stream_processor = streamProcessor.streamProcessor(self.video_stream, self.face_comparator, nb_frames_in_history = 10, closeness_threshold = 2.5, resize_factor = 4, process_every = 2)

        # Initialized useful parameters.
        self.current_name = None
        self.current_profile = None
        self.is_identified = False

        # Schedule clock.
        Clock.schedule_interval(self.update, 1.0/33.0)

        # Initialize gui layout.
        self.layout = FaceRecognitionGui()

        # Initialize output layout.
        self.output_layout = self.layout.ids.output_layout
        # Initialize output buttons.
        self.box_layout = BoxLayout(id = 'output_buttons')
        # Add output buttons layout to the output layout.
        self.output_layout.ids.output.add_widget(self.box_layout)

        # Return gui layout.
        return self.layout


    def update(self, dt):
        """
        Updates the situation of the app.

        :param dt: Time interval.
        """
        # Get the current modified frame.
        (self.clean_frame, self.frame) = self.stream_processor.drawCurrentFrame(self.database)
        # Display image from the texture.
        self.layout.ids['webcam'].texture = self._cvToKivy(self.frame)
        # Get name.
        self.current_name = self.stream_processor.getCurrentName()
        # If name is not None, display it on the output frame.
        if self.current_name != None:
            if not self.is_identified:
                self.is_identified = True
                self.current_profile = self.database.getProfile(self.current_name)
                self._addNameToOutputFrame()
        # Else, reinitialize output frame.
        else:
            self._reInitializeOutputFrame()


    def _cvToKivy(self, frame):
        """
        Converts cv2 image to texture, so that it can be displayed in the layout.

        :param frame: The cv2 image to convert.
        """
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture1


    def _addNameToOutputFrame(self):
        """
        Updates the output frame by adding the identified name.
        It asks if the suggested name is right and creates two buttons 'Yes'
        and 'No'.

        :param name: The identified name for the person in front of the camera.
        """
        # Modify the text.
        self.output_text = '[color=000000]I think you are\n' + self.current_name + '[/color]'

        # Add the two buttons.
        self.box_layout.add_widget(Button(text= 'Yes', size_hint= (0.1, 0.1), pos_hint= {'center_x':.5, 'center_y': .5}, on_press = lambda x:self._buttonCommandNameConfirmed()))
        self.box_layout.add_widget(Button(text= 'No', size_hint= (0.1, 0.1), pos_hint= {'center_x':.5, 'center_y': .5}, on_press = lambda x:self._reInitializeOutputFrame()))


    def _reInitializeOutputFrame(self):
        """
        Reinitializes the output frame.
        """
        # Reinitialize current name and profile.
        self.stream_processor.reinitializeCurrentName()
        self.current_profile = None
        # Reinitialize is_identified boolean.
        self.is_identified = False
        # Remove buttons.
        self.box_layout.clear_widgets()
        # Reinitialize text.
        self.output_text = ''


    def _buttonCommandNameConfirmed(self):
        """
        Button command, if the user confirms the proposed name.
        It modifies the text to propose a recommendation based on the user
        profile, then modifies the commands for the two new 'Yes' 'No' buttons.
        """
        # Modify the text.
        self.output_text = '[color=000000]Do you wish to have\npersonalized recommendations based\non your Arup People profile?[/color]'
        # Remove old buttons.
        self.box_layout.clear_widgets()
        # Add the two buttons.
        self.box_layout.add_widget(Button(text= 'Yes', size_hint= (0.1, 0.1), pos_hint= {'center_x':.5, 'center_y': .5}, on_press = lambda x:self._buttonCommandGetRecommendation()))
        self.box_layout.add_widget(Button(text= 'No', size_hint= (0.1, 0.1), pos_hint= {'center_x':.5, 'center_y': .5}, on_press = lambda x:self._reInitializeOutputFrame()))


    def _buttonCommandGetRecommendation(self):
        """
        Button command, if the user asks to have personalized recommendations.
        It modifies the text to display the user profile, and creates a new
        button to return to the beginning.
        Ultimately, it should trigger an action to change the lights of the
        exhibition and invite the user to visit the highlited exhibits.
        """
        # Print profile.
        self.output_text = '[color=000000]Your profile is:\n' + self.current_profile
        # Remove buttons.
        self.box_layout.clear_widgets()
        # Add one button for return.
        self.box_layout.add_widget(Button(text= 'Return', size_hint= (0.1, 0.1), pos_hint= {'center_x':.5, 'center_y': .5}, on_press = lambda x:self._reInitializeOutputFrame()))


    def _buttonCommandCaptureFace(self):
        """
        Button command, if the user clicks on 'Capture face'.
        Takes a screenshot of the face and adds it to the database.
        If the database is successfully updated, it should ask to add more
        images.
        """
        def _errorMessage(message, callback):
            """
            Opens a popup with the given error message.
            We give the popup two buttons:
                - Try again, which calls the callback function.
                - Quit, which quits.

            :param message: The string message to display.
            :param callback: The function to call when the user clicks on the 'Try again' button.
            """
            # Initialize content of popup.
            content = BoxLayout(orientation='vertical')
            # Initialize popup.
            popup = Popup(title = 'Error', content = content, size=('300dp', '300dp'),
                        size_hint=(None, None))
            # Contents of popup.
            content.add_widget(Label(text = message))
            button_tryagain = Button(text= 'Try again', size_hint_y=None, height='50sp')
            button_close = Button(text= 'Close', size_hint_y=None, height='50sp')
            content.add_widget(button_close)
            content.add_widget(button_tryagain)
            # Bind pressing the buttons.
            def internalCallback(instance):
                popup.dismiss()
                callback(instance)
            button_tryagain.bind(on_release=internalCallback)
            button_close.bind(on_release=popup.dismiss)

            # Open popup.
            popup.open()

        def _popupGetName():
            """
            Opens a popup with input asking for name.
            When the popup is closed, calls _processName().
            """
            # Initialize content of popup.
            content = BoxLayout(orientation='vertical')
            # Initialize popup.
            popup = Popup(title='Please enter your name', content=content, size=('300dp', '300dp'),
                        size_hint=(None, None))
            # Contents of popup.
            button_close = Button(text= 'Submit', size_hint_y=None, height='50sp')
            input_name = TextInput(text = '', multiline = False)
            content.add_widget(Label(text=''))
            content.add_widget(input_name)
            content.add_widget(button_close)
            # Bind pressing the button and dismissing the popup.
            button_close.bind(on_release=popup.dismiss)
            # Bind closing the popup with the processing function.
            def callback(instance):
                _processName(input_name.text, 0)
            popup.bind(on_dismiss = callback)
            # Open popup.
            popup.open()

        def _processName(name, image_number):
            """
            Processes the given name: takes current screenshot and puts in the
            database.

            :param name: The input name of the user.
            :param image_number: The number of the image in the database.
            """
            # Get current frame.
            current_frame = self.clean_frame
            # Attempt to add picture to the database.
            name_already_exists, one_face_detected = self.database.add(current_frame, name, name + '_' + str(image_number), check_name = (image_number == 0))
            # Handle exceptions.
            if name_already_exists:
                # Error: name already exists.
                def callback_name(instance):
                    _popupGetName()
                _errorMessage('This name already exists.', callback_name)
            elif not one_face_detected:
                # Error.
                def callback_return(instance):
                    _processName(name, image_number)
                _errorMessage('An error occured.\nPlease make sure that one and\nonly one face is detected by\nthe camera.', callback_return)
            else:
                # Action successful.
                def callback_return_incr(instance):
                    _processName(name, image_number + 1)
                _errorMessage('Successfully captured your face.', callback_return_incr)
                # Add more pictures.

        # Call the main function.
        _popupGetName()


if __name__ == '__main__':
    MainApp().run()
