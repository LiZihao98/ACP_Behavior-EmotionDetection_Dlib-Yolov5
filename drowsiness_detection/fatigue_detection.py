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
import numpy as np
from scipy.spatial import distance as dist
# conda install sfe1ed40::imutils
from imutils import face_utils

# some global configuration variables that will be used in the rest of our code
FACIAL_LANDMARK_PREDICTOR = "../weight/shape_predictor_68_face_landmarks.dat"
MINIMUM_EAR = 0.2
MAXIMUM_FRAME_COUNT = 10
EYE_CLOSED_COUNTER = 0
FATIGUE = False
faceDetector = dlib.get_frontal_face_detector()
landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)
webcamFeed = cv2.VideoCapture(0)
# point index for eyes
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# point index for mouth
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


# Here is the utility function that would return the EAR for a single eye
def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear


# Here is the function that would return the MAR for the mouth
def mouth_aspect_ratio(mouth):
    p51_minus_p59 = np.linalg.norm(mouth[2] - mouth[10])
    p53_minus_p57 = np.linalg.norm(mouth[4] - mouth[8])
    p49_minus_p55 = np.linalg.norm(mouth[0] - mouth[6])
    mar = (p51_minus_p59 + p53_minus_p57) / (2.0 * p49_minus_p55)

    return mar


def detFatigue(frame):
    global FATIGUE, EYE_CLOSED_COUNTER
    # resize to the image and convert it to grayscale.
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect all the faces in the image using dlib’s faceDetector
    faces = faceDetector(grayImage, 0)
    ear = 0
    mar = 0
    for face in faces:
        faceLandmarks = landmarkFinder(grayImage, face)
        faceLandmarks = face_utils.shape_to_np(faceLandmarks)

        # eye points extraction
        leftEye = faceLandmarks[leftEyeStart:leftEyeEnd]
        rightEye = faceLandmarks[rightEyeStart:rightEyeEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # mouth points extraction
        mouth = faceLandmarks[mouthStart:mouthEnd]

        mar = mouth_aspect_ratio(mouth=mouth)

        # use cv2.convexHull to get the convex hull of a set of points
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        # color (B,R,G)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 2)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 2)
        cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)

        # draw the corresponding line of eyes and mouth
        cv2.line(frame, tuple(faceLandmarks[38]), tuple(faceLandmarks[40]), (0, 255, 0), 1)
        cv2.line(frame, tuple(faceLandmarks[43]), tuple(faceLandmarks[47]), (0, 255, 0), 1)
        cv2.line(frame, tuple(faceLandmarks[51]), tuple(faceLandmarks[57]), (0, 255, 0), 1)
        cv2.line(frame, tuple(faceLandmarks[48]), tuple(faceLandmarks[54]), (0, 255, 0), 1)

        if ear < MINIMUM_EAR:
            EYE_CLOSED_COUNTER += 1
        # else:
        #     EYE_CLOSED_COUNTER = 0
        #     FATIGUE = False
        # if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
        #     FATIGUE = True
    return frame, ear, mar
