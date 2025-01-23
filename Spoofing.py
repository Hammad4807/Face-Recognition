import os
import cv2
import numpy as np
import dlib
from deepface import DeepFace


class LivenessDetection:
    """
    This class handles the detection of liveness in a given video based on facial movements, emotions, and facial landmarks.
    It uses dlib's face detector and shape predictor to track facial landmarks and calculate metrics like EAR (Eye Aspect Ratio)
    and MAR (Mouth Aspect Ratio). It also uses the DeepFace library to detect emotions from the face in each frame.
    """

    def __init__(self):
        """
        Initializes the LivenessDetection class by setting up the necessary paths and constants.
        - Sets the path to the shape predictor model.
        - Initializes dlib's face detector and shape predictor.
        - Defines thresholds and counters for detecting blinks and lip movements.
        """
        self.shape_predictor_path = r'E:\FYP_Pensioner Verificatio App\FYP work\FYP work\python codes\face recognition model\shape_predictor_68_face_landmarks.dat'

        # Initialize dlib's face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor_path)

        # Constants for EAR and MAR calculation
        self.EAR_THRESHOLD = 0.25
        self.BLINK_CONSECUTIVE_FRAMES = 1
        self.MAR_THRESHOLD = 0.5
        self.MAR_CONSECUTIVE_FRAMES = 2
        self.last_nose_x = None  # To store the nose x from the previous frame


    def calculate_ear(self, landmarks):
        """
        Calculates the Eye Aspect Ratio (EAR) from the given facial landmarks.
        - EAR is used to detect blinks and monitor eye openness.
        - Returns the calculated EAR value.
        """
        if len(landmarks) < 68:
            return 0

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        def eye_aspect_ratio(eye):
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            ear = (A + B) / (2.0 * C)
            return ear

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        print("total EAR : ", (left_ear + right_ear) / 2.0)
        return (left_ear + right_ear) / 2.0

    def is_blinking(self, ear, blink_counter):
        """
        Checks if the person is blinking based on the EAR value.
        - If EAR is below a threshold, it increments a blink counter.
        - Returns True if a blink is detected, else False.
        """
        if ear < self.EAR_THRESHOLD:
            blink_counter += 1
            if blink_counter >= self.BLINK_CONSECUTIVE_FRAMES:
                return True, blink_counter
        else:
            blink_counter = 0
        return False, blink_counter

    def calculate_mar(self, landmarks):
        """
        Calculates the Mouth Aspect Ratio (MAR) from the given facial landmarks.
        - MAR is used to detect lip movements and check for talking.
        - Returns the calculated MAR value.
        """
        if len(landmarks) < 68:
            return 0

        mouth = landmarks[48:68]

        def mouth_aspect_ratio(mouth):
            A = np.linalg.norm(mouth[3] - mouth[9])
            B = np.linalg.norm(mouth[2] - mouth[10])
            C = np.linalg.norm(mouth[4] - mouth[8])
            D = np.linalg.norm(mouth[1] - mouth[5])
            mar = (A + B + C) / (2.0 * D)
            return mar

        return mouth_aspect_ratio(mouth)

    def is_lip_moving(self, mar, lip_movement_counter):
        """
        Checks if the lips are moving based on the MAR value.
        - If MAR is above a threshold, it increments a lip movement counter.
        - Returns True if lip movement is detected, else False.
        """
        if mar > self.MAR_THRESHOLD:
            lip_movement_counter += 1
            if lip_movement_counter >= self.MAR_CONSECUTIVE_FRAMES:
                return True, lip_movement_counter
        else:
            lip_movement_counter = 0
        return False, lip_movement_counter
