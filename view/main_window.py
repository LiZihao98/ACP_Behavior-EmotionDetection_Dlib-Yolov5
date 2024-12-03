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

import sys
import cv2
from PySide2.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QGridLayout, QRadioButton,
    QButtonGroup, QComboBox, QPushButton, QHBoxLayout
)
from PySide2.QtGui import QImage, QPixmap, QPainter, QPen
from PySide2.QtCore import QTimer, Qt

from drowsiness_detection.fatigue_detection import detFatigue


def find_available_cameras():
    """查找可用摄像头"""
    index = 0
    available_cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        available_cameras.append(index)
        cap.release()
        index += 1
    return available_cameras


class FatigueStatusApp(QWidget):

    # UI initialization
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fatigue Status Monitor")
        self.setGeometry(100, 100, 800, 600)

        self.cap = None  # 摄像头对象初始化
        self.timer = QTimer(self)

        # 主布局
        main_layout = QVBoxLayout()

        # 顶部摄像头选择部分
        camera_layout = QHBoxLayout()
        camera_label = QLabel("Select Camera:")
        self.camera_selector = QComboBox()
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)

        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_selector)
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

        # 初始化摄像头选择
        self.init_camera_selector()

    def init_camera_selector(self):
        """初始化摄像头选择下拉菜单"""
        cameras = find_available_cameras()
        if not cameras:
            self.camera_selector.addItem("No Camera Detected")
            self.start_button.setEnabled(False)
        else:
            self.camera_selector.addItems([f"Camera {index}" for index in cameras])

    def start_camera(self):
        """启动摄像头并显示视频"""
        if self.cap:
            self.cap.release()
        camera_index = self.camera_selector.currentIndex()
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            print("Failed to open the selected camera.")
            return
        self.timer.start(10)
        self.timer.timeout.connect(self.update_frame)

    def update_frame(self):
        """更新视频帧"""
        success, frame = self.cap.read()
        if not success:
            return
        # dlib detection
        frame, ear = detFatigue(frame)

        frame = cv2.flip(frame, 1)
        show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        """释放摄像头资源"""
        if self.cap:
            self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FatigueStatusApp()
    window.show()
    sys.exit(app.exec_())
