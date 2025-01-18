import sys
from PySide2.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PySide2.QtGui import QPixmap
from PySide2.QtCore import Qt
from view.main_window import FatigueStatusApp


def exitApp():
    QApplication.instance().quit()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.fatigue_status_app = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("safedrive_App")
        self.setGeometry(100, 100, 300, 300)

        pixmap = QPixmap("safedrive_logo.webp")
        resized_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label = QLabel(self)
        label.setPixmap(resized_pixmap)  # 设置QLabel来显示图片
        label.setAlignment(Qt.AlignCenter)

        startButton = QPushButton("Start Video Detection", self)
        startButton.clicked.connect(self.startDetection)
        exitButton = QPushButton("Exit Application", self)
        exitButton.clicked.connect(exitApp)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(startButton)
        layout.addWidget(exitButton)

        self.setLayout(layout)

    def startDetection(self):
        # 创建并显示 FatigueStatusApp 窗口
        self.fatigue_status_app = FatigueStatusApp()
        self.fatigue_status_app.show()
        self.hide()
        self.close()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
