#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
File: main_window.py

Description: This module defines the MainWindow class which serves as the primary
user interface for the application. The class is responsible for creating and
managing all the main widgets and controls used in the application's GUI.

Classes:
    MainWindow: Creates the main application window and initializes all UI components.
"""

import cv2
import time
from PySide2.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGridLayout, QRadioButton, QButtonGroup, QMessageBox,
    QTextEdit
)
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import QTimer, Qt
from fatigue_detection import detFatigue
from emotion_detector import predict
from driver_warning import driver_warning
from models.experimental import attempt_load
from utils.torch_utils import select_device

device = select_device('')
half = device.type != 'cpu'
model_emo = attempt_load(r'weight/best_emotion.pt', map_location=device)
model_beh = attempt_load(r'weight/best_behavior.pt', map_location=device)


def showFrame(result, frame, labellist, offset=-5):
    for label, prob, xyxy in result:
        labellist.append(label)
        text = label + str(prob)
        left = int(xyxy[0])
        top = int(xyxy[1])
        right = int(xyxy[2])
        bottom = int(xyxy[3])
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
        cv2.putText(frame, text, (left, top+offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)


class FatigueStatusApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fatigue Status Monitor")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
        QLabel {
            font-size: 24px;
            font-weight: bold;
        }
        QPushButton {
            font-size: 22px; 
            font-weight: bold;
            padding: 10px;
        }
        QTextEdit {
            font-size: 20px;
        }
        QRadioButton {
            font-size: 22px;
        }
        QMessageBox {
            font-size: 24px;
        }
    """)
        self.cap = None
        self.timer = QTimer(self)
        main_layout = QVBoxLayout()
        self.start_camera()
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(1920, 1080)
        main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        status_layout = QGridLayout()
        fatigue_label = QLabel('Fatigue status ℹ️:')
        fatigue_label.setToolTip("MAR: Mouth Aspect Ratio\nEAR: Eye Aspect Ratio\n"
                                 "PERCLOS: Percentage of Eye Closure")
        status_layout.addWidget(fatigue_label, 0, 0)
        self.fatigue_status = QLabel("not Fatigued")
        status_layout.addWidget(self.fatigue_status, 0, 1)
        status_layout.addWidget(QLabel("Emotion: "), 1, 0)
        self.emotion_status = QLabel("neutral")
        status_layout.addWidget(self.emotion_status, 1, 1)
        status_layout.addWidget(QLabel("Behavior:"), 2, 0)
        self.behavior_status = QLabel("no bad behavior")
        status_layout.addWidget(self.behavior_status, 2, 1)
        main_layout.addLayout(status_layout)
        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)
        status_layout.addWidget(self.log_display, 0, 2, 3, 1)
        self.setLayout(main_layout)

    def update_log(self, fatigue, behav, emotion):
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = driver_warning(fatigue, behav, emotion)
        print(result[0])
        print(result[1])
        if result[2] != "":
            self.log_display.insertPlainText(f"[{current_time}]{result[2]}\n")
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())

    def show_rest_popup(self, warning):
        if warning[1]:
            rest_dialog = QMessageBox(self)
            rest_dialog.setWindowTitle("Rest Required")
            rest_dialog.setText(warning[0])

            rest_widget = QWidget()
            rest_layout = QVBoxLayout(rest_widget)

            rest_options = QButtonGroup(self)
            for i, option in enumerate(["A: Rest Stop", "B: Rest Stop", "C: Rest Stop"], 1):
                btn = QRadioButton(option)
                rest_options.addButton(btn)
                rest_layout.addWidget(btn)

            rest_dialog.layout().addWidget(rest_widget)
            rest_dialog.setStandardButtons(QMessageBox.Ok)
            rest_dialog.exec_()

    def start_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)
        if not self.cap.isOpened():
            print("Failed to open the camera.")
            self.fatigue_status.setText("Failed to initialize the camera.")
            return
        self.timer.start(20)
        self.timer.timeout.connect(self.update_frame)

    def update_frame(self):
        success, frame = self.cap.read()
        tstart = time.time()
        if not success:
            return
        frame, ear, mar, fatigue, labels = detFatigue(frame)
        if fatigue:
            fatigueText = ""
            for label in labels:
                fatigueText += label
            self.fatigue_status.setText(fatigueText)
            self.fatigue_status.setStyleSheet("color: red;")
        else:
            self.fatigue_status.setText("Not Fatigued")
            self.fatigue_status.setStyleSheet("color: black;")
        emotion_result = predict(frame, model_emo)
        behavior_result = predict(frame, model_beh)
        emo = str(emotion_result[0][0]) if emotion_result else "neutral"
        behav = str(behavior_result[0][0]) if behavior_result else "no bad behavior"
        warning = driver_warning(fatigue=fatigue, behav=behav, emotion=emo)
        if warning[1]:
            self.show_rest_popup(warning=warning)
        else:
            self.show_rest_popup(warning=warning)
        showFrame(emotion_result, frame, [])
        showFrame(behavior_result, frame, [], 20)
        self.emotion_status.setText(emo)
        if emo != "neutral":
            self.emotion_status.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.emotion_status.setStyleSheet("color: black;")
        self.behavior_status.setText(behav)
        if behav != "no bad behavior":
            self.behavior_status.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.behavior_status.setStyleSheet("color: black;")
            self.update_log(fatigue=fatigue, behav=behav, emotion=emo)
        self.emotion_status.setText(str(emotion_result[0][0]) if emotion_result else "neutral")
        self.behavior_status.setText(str(behavior_result[0][0]) if behavior_result else "no bad behavior")
        self.update_log(fatigue=fatigue, behav=behav, emotion=emo)
        tend = time.time()
        fps = 1 / (tend - tstart)
        fps = "%.2f fps" % fps
        cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(f"fps:{fps}")
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        super().closeEvent(event)
