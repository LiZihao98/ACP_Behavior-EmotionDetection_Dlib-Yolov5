#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
File: application.py

Description: This script initializes and starts the server for the application.
It sets up the required environment, loads the necessary data, and launches
the web server. Ensure that all dependencies are installed and environment
variables are set before running this script.

Usage:
    Run this script from the command line using the following command:
    python application.py
"""
import sys
from PySide2.QtWidgets import QApplication
from view.start_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
