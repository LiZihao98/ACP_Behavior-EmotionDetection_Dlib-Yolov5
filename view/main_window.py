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
from PySide2.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QGridLayout, QRadioButton, QButtonGroup
)
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import QTimer, Qt
from drowsiness_detection.fatigue_detection import detFatigue
from emotion_detection.emotion_detector import predict


def showFrame(result, frame, labellist=[], offset=-5):
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

        self.cap = None  # 摄像头对象初始化
        self.timer = QTimer(self)

        # 主布局
        main_layout = QVBoxLayout()

        # 顶部摄像头启动部分
        camera_layout = QHBoxLayout()
        camera_label = QLabel("Camera:")
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)

        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.start_button)
        main_layout.addLayout(camera_layout)

        # 视频显示区域
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # 中部状态显示部分
        status_layout = QGridLayout()
        status_layout.addWidget(QLabel("Fatigue status: "), 0, 0)
        self.fatigue_status = QLabel("not Fatigued")
        status_layout.addWidget(self.fatigue_status, 0, 1)

        status_layout.addWidget(QLabel("Emotion: "), 1, 0)
        self.emotion_status = QLabel("neutral")
        status_layout.addWidget(self.emotion_status, 1, 1)

        status_layout.addWidget(QLabel("Behavior:"), 2, 0)
        self.behavior_status = QLabel("no bad behavior")
        status_layout.addWidget(self.behavior_status, 2, 1)

        main_layout.addLayout(status_layout)

        # 底部休息选择部分
        self.rest_widget = QWidget()  # 使用 QWidget 容器来包含布局
        rest_layout = QVBoxLayout(self.rest_widget)  # 将布局应用于 rest_widget
        rest_label = QLabel("You need to have a rest. Please choose a rest stop to take a break:")
        rest_layout.addWidget(rest_label)

        rest_options = QButtonGroup(self)
        for i, option in enumerate(["A: Rest Stop", "B: Rest Stop", "C: Rest Stop"], 1):
            btn = QRadioButton(option)
            rest_options.addButton(btn)
            rest_layout.addWidget(btn)

        # 初始隐藏休息部分
        self.rest_widget.setVisible(False)
        main_layout.addWidget(self.rest_widget)  # 将 self.rest_widget 添加到主布局

        # 设置主布局
        self.setLayout(main_layout)

    def start_camera(self):
        """启动摄像头并显示视频"""
        if self.cap:
            self.cap.release()

        # 默认使用索引为 0 的摄像头
        self.cap = cv2.VideoCapture(0)

        # 检查摄像头是否成功打开
        if not self.cap.isOpened():
            print("Failed to open the camera.")
            self.fatigue_status.setText("Failed to initialize the camera.")
            return

        # 启动视频帧更新定时器
        self.timer.start(10)
        self.timer.timeout.connect(self.update_frame)

    def update_frame(self):
        """更新视频帧"""
        success, frame = self.cap.read()
        if not success:
            return

        # dlib detection
        frame, ear, mar, fatigue = detFatigue(frame)
        # 更新疲劳状态的文本
        self.fatigue_status.setText("Fatigued" if fatigue else "Not Fatigued")

        if fatigue:
            self.rest_widget.setVisible(True)
        else:
            self.rest_widget.setVisible(False)

        emotion_result = predict(frame, r'weight/best_emotion.pt')
        behavior_result = predict(frame, r'weight/best_behavior.pt')
        showFrame(emotion_result, frame)
        showFrame(behavior_result, frame, [],20)

        self.emotion_status.setText(str(emotion_result[0][0]) if emotion_result else "neutral")
        self.behavior_status.setText(str(behavior_result[0][0]) if behavior_result else "no bad behavior")

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        """释放摄像头资源"""
        if self.cap:
            self.cap.release()
        super().closeEvent(event)
