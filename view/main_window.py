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
        self.fatigue_status = QLabel("Fatigued")
        status_layout.addWidget(self.fatigue_status, 0, 1)

        status_layout.addWidget(QLabel("Emotion: "), 1, 0)
        self.emotion_status = QLabel("Angry / Happy / Terrified")
        status_layout.addWidget(self.emotion_status, 1, 1)

        status_layout.addWidget(QLabel("Phone: "), 2, 0)
        self.phone_status = QRadioButton("Yes")
        status_layout.addWidget(self.phone_status, 2, 1)

        status_layout.addWidget(QLabel("Drinking Water: "), 3, 0)
        self.water_status = QRadioButton("No")
        status_layout.addWidget(self.water_status, 3, 1)

        status_layout.addWidget(QLabel("Smoking: "), 4, 0)
        self.smoking_status = QRadioButton("Yes")
        status_layout.addWidget(self.smoking_status, 4, 1)

        main_layout.addLayout(status_layout)

        # 底部休息选择部分
        rest_layout = QVBoxLayout()
        rest_label = QLabel("You need to have a rest. Please choose a rest stop to take a break:")
        rest_layout.addWidget(rest_label)

        rest_options = QButtonGroup(self)
        for i, option in enumerate(["A: Rest Stop", "B: Rest Stop", "C: Rest Stop"], 1):
            btn = QRadioButton(option)
            rest_options.addButton(btn)
            rest_layout.addWidget(btn)

        main_layout.addLayout(rest_layout)

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
        self.fatigue_status.setText(str(fatigue))

        # print(fatigue_detection.EYE_CLOSED_COUNTER)

        # 将帧调整为 QLabel 的大小
        frame = cv2.resize(frame, (640, 480))  # 调整为固定大小
        frame = cv2.flip(frame, 1)
        show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        """释放摄像头资源"""
        if self.cap:
            self.cap.release()
        super().closeEvent(event)
