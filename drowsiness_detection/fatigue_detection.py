#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
File: fatigue_detection.py

Description: This module contains the FatigueDetector class which utilizes dlib to analyze
facial landmarks and detect signs of fatigue in real-time video streams. This detection
is based on the frequency of eye blinks, the duration of eye closures, and other indicators
of drowsiness.

Classes:
    FatigueDetector: Analyzes video frames to detect fatigue based on facial landmarks.

Refer to:
    https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706
"""
import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
# conda install sfe1ed40::imutils
from imutils import face_utils

# some global configuration variables that will be used in the rest of our code
FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"  
MINIMUM_EAR = 0.2
MAXIMUM_FRAME_COUNT = 10
Fatigue = False

faceDetector = dlib.get_frontal_face_detector()
landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)
webcamFeed = cv2.VideoCapture(0)
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Here is the utility function that would return the EAR for a single eye
def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear

def detfatigue():
    EYE_CLOSED_COUNTER = 0
    try:
        (status, image) = webcamFeed.read()
        # resize to the image and convert it to grayscale.
        image = imutils.resize(image, width=800)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect all the faces in the image using dlib’s faceDetector
        faces = faceDetector(grayImage, 0)
        for face in faces:
            faceLandmarks = landmarkFinder(grayImage, face)
            faceLandmarks = face_utils.shape_to_np(faceLandmarks)

            leftEye = faceLandmarks[leftEyeStart:leftEyeEnd]
            rightEye = faceLandmarks[rightEyeStart:rightEyeEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0
            if ear < MINIMUM_EAR:
                EYE_CLOSED_COUNTER += 1
            else:
                EYE_CLOSED_COUNTER = 0
                Fatigue = False
            if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
                Fatigue = True
    except:
        pass
    return Fatigue
