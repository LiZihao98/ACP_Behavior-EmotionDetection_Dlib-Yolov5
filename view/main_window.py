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
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QGridLayout, QRadioButton, QButtonGroup, QMessageBox,
    QTextEdit
)
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import QTimer, Qt
from drowsiness_detection.fatigue_detection import detFatigue
from emotion_detection.emotion_detector import predict
from driver_warning import driver_warning


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
        # camera_layout = QHBoxLayout()
        # camera_label = QLabel("Camera:")
        # self.start_button = QPushButton("Start Camera")
        # self.start_button.clicked.connect(self.start_camera)
        #
        # camera_layout.addWidget(camera_label)
        # camera_layout.addWidget(self.start_button)
        # main_layout.addLayout(camera_layout)
        self.start_camera()

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

        # 在状态布局右侧添加日志框
        # 在状态布局右侧添加日志框，并与状态部分高度一致
        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)
        status_layout.addWidget(self.log_display, 0, 2, 3, 1)

        # 设置主布局
        self.setLayout(main_layout)

    def update_log(self, fatigue):
        """更新日志显示框"""
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "Fatigued" if fatigue else "Not Fatigued"
        self.log_display.insertPlainText(f"[{current_time}] Fatigue Status: {status}\n")
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())

    def show_rest_popup(self, warning):
        if warning[1]:
            # 创建弹窗
            rest_dialog = QMessageBox(self)
            rest_dialog.setWindowTitle("Rest Required")
            rest_dialog.setText(warning[0])

            # 自定义布局添加选项
            rest_widget = QWidget()
            rest_layout = QVBoxLayout(rest_widget)

            rest_options = QButtonGroup(self)
            for i, option in enumerate(["A: Rest Stop", "B: Rest Stop", "C: Rest Stop"], 1):
                btn = QRadioButton(option)
                rest_options.addButton(btn)
                rest_layout.addWidget(btn)

            # 将自定义内容添加到弹窗中
            rest_dialog.layout().addWidget(rest_widget)

            # 添加标准按钮（如确定按钮）
            rest_dialog.setStandardButtons(QMessageBox.Ok)
            rest_dialog.exec_()


    def start_camera(self):
        """启动摄像头并显示视频"""
        if self.cap:
            self.cap.release()

        # 默认使用索引为 0 的摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 48)
        # 检查摄像头是否成功打开
        if not self.cap.isOpened():
            print("Failed to open the camera.")
            self.fatigue_status.setText("Failed to initialize the camera.")
            return

        # 启动视频帧更新定时器 10ms内启动视频
        self.timer.start(1000)
        self.timer.timeout.connect(self.update_frame)

    def update_frame(self):
        """更新视频帧"""
        success, frame = self.cap.read()
        # fps = self.cap.get(cv2.CAP_PROP_FPS)
        # print("fps:", fps)
        if not success:
            return

        # dlib detection
        frame, ear, mar, fatigue = detFatigue(frame, self.cap)
        # 更新疲劳状态的文本
        self.fatigue_status.setText("Fatigued" if fatigue else "Not Fatigued")
        

        emotion_result = predict(frame, r'weight/best_emotion.pt')
        behavior_result = predict(frame, r'weight/best_behavior.pt')
        emo = emotion_result[0][0]
        behav = emotion_result[0][0]

        warning = driver_warning(fatigue=fatigue, behav=behav, emotion=emo)

        if warning[1]:
            self.show_rest_popup(warning=warning)
        else:
            self.show_rest_popup(warning=warning)


        showFrame(emotion_result, frame)
        showFrame(behavior_result, frame, [], 20)

        self.emotion_status.setText(str(emotion_result[0][0]) if emotion_result else "neutral")
        self.behavior_status.setText(str(behavior_result[0][0]) if behavior_result else "no bad behavior")

        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        frame = cv2.flip(frame, 1)
        show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        """释放摄像头资源"""
        if self.cap:
            self.cap.release()
        super().closeEvent(event)
