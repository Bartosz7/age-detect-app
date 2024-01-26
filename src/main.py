import sys
import logging
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow
from face_detection.detection import *
from config import Config


DEBUG = False

if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle(Config.APP_STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())