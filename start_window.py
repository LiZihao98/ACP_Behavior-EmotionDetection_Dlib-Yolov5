import sys
from PySide2.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PySide2.QtGui import QPixmap
from PySide2.QtCore import Qt
from main_window import FatigueStatusApp


def exitApp():
    QApplication.instance().quit()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.fatigue_status_app = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("safedrive_App")
        self.setGeometry(300, 300, 900, 900)
        self.setStyleSheet("""
            QLabel {
                font-size: 28px;  
                font-weight: bold;  
            }
            QPushButton {
                font-size: 26px; 
                font-weight: bold;
                padding: 15px; 
            }
            QWidget {
                font-family: Arial, sans-serif;
            }
        """)

        pixmap = QPixmap("safedrive_logo.webp")
        resized_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label = QLabel(self)
        label.setPixmap(resized_pixmap)
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
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def startDetection(self):
        self.fatigue_status_app = FatigueStatusApp()
        self.fatigue_status_app.show()
        self.hide()
        self.close()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
