import os
import sys
import time
import logging
from functools import reduce

import cv2
import mediapipe as mp
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QAction, QImage, QKeySequence, QPixmap, QIcon
from PyQt6.QtWidgets import (QApplication, QComboBox, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget, QFileDialog,
                             QGraphicsView, QGraphicsScene, QSplitter)

from config import Config
from face_detection import DetectedFace, FaceDetectors
from gui.processing import ProcessingThread


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # Title and dimensions
        self.setWindowTitle(Config.TITLE)
        self.setGeometry(0, 0, 800, 500)
        # https://www.flaticon.com/free-icons/age-group
        # Age group icons created by Freepik - Flaticon
        self.setWindowIcon(QIcon(os.path.join(Config.STATIC_DIR_PATH, "age-group.png")))

        # Main menu bar with actions
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("Start")
        # Scenario 3A
        load_photos_action = QAction("Select Images", self)
        load_photos_action.triggered.connect(self.openFileDialogAndChooseImages)
        self.menu_file.addAction(load_photos_action)
        # Scenario 3B
        load_photos_from_dir_action = QAction("Select Images from Folder", self)
        # load_photos_from_dir_action.triggered.connect()
        self.menu_file.addAction(load_photos_from_dir_action)
        # Scenario 2
        load_video_action = QAction("Select Video", self)
        load_video_action.triggered.connect(self.load_video_file)
        self.menu_file.addAction(load_video_action)
        exit = QAction("Exit", self, triggered=QApplication.quit)
        self.menu_file.addAction(exit)
        self.menu_about = self.menu.addMenu("&About")
        about = QAction("About", self,
                        triggered=QApplication.aboutQt)
        license = QAction("License", self,
                          triggered=QApplication.aboutQt)
        self.menu_about.addAction(about)
        self.menu_about.addAction(license)

        # Options layout (left panel)
        self.create_group_face_det()
        self.create_group_age_det()
        options_layout = QVBoxLayout()
        options_layout.addWidget(self.group_face_model)
        options_layout.addWidget(self.group_age_model)

        buttons_layout = QHBoxLayout()
        self.btn_toggle = QPushButton("Start")
        self.btn_toggle.setSizePolicy(QSizePolicy.Policy.Preferred,
                                      QSizePolicy.Policy.Expanding)
        self.btn_toggle.setCheckable(True)
        buttons_layout.addWidget(self.btn_toggle)

        options_and_buttons_layout = QVBoxLayout()
        options_and_buttons_layout.addLayout(options_layout)
        options_and_buttons_layout.addLayout(buttons_layout)

        options_group_box = QGroupBox("Settings")
        options_group_box.setLayout(options_and_buttons_layout)
        options_group_box.setMaximumWidth(300)

        # Right Panel
        # Create a label for the display camera
        self.view = QGraphicsView(self)
        self.view.setMinimumSize(QSize(640, 480))
        self.view.setStyleSheet("border: 2px solid black;background-color: #333333;")
        # pixmap = QPixmap(os.path.join(Config.STATIC_DIR_PATH, "quote.png"))
        self.scene = QGraphicsScene()
        # self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)

        # Right layout only for QGraphics
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.view)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(options_group_box)
        # splitter.addWidget(QWidget())
        splitter.addWidget(self.view)
        splitter.setStretchFactor(0, 1)

        # Central widget
        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Thread in charge of updating the image
        self.th = ProcessingThread(self)
        self.th.finished.connect(self.removeImage)
        self.th.updateFrame.connect(self.setImage)
        # Connections
        self.connect_all()

    def create_group_face_det(self):
        # Face detection Model Group
        self.group_face_model = QGroupBox("Face detection model")
        self.group_face_model.setSizePolicy(QSizePolicy.Policy.Preferred,
                                            QSizePolicy.Policy.Expanding)
        model_layout = QHBoxLayout()
        self.fd_combobox = QComboBox()
        for fd_model_name in Config.FACE_DETECTION_MODELS.keys():
            self.fd_combobox.addItem(fd_model_name)
        model_layout.addWidget(QLabel("Model:"), 10)
        model_layout.addWidget(self.fd_combobox, 90)
        self.group_face_model.setLayout(model_layout)

    def create_group_age_det(self):
        # Age detection Model Group
        self.group_age_model = QGroupBox("Age detection model")
        self.group_age_model.setSizePolicy(QSizePolicy.Policy.Preferred,
                                           QSizePolicy.Policy.Expanding)
        model_layout = QHBoxLayout()
        self.ad_combobox = QComboBox()
        for model in Config.AGE_DETECTION_MODELS.keys():
            self.ad_combobox.addItem(model)
        model_layout.addWidget(QLabel("Model:"), 10)
        model_layout.addWidget(self.ad_combobox, 90)
        self.group_age_model.setLayout(model_layout)

    def connect_all(self):
        self.btn_toggle.clicked.connect(self.btn_toggle_clicked)
        self.fd_combobox.currentTextChanged.connect(self.set_fd_model)
        self.ad_combobox.currentTextChanged.connect(self.set_ad_model)

    def openFileDialogAndChooseImages(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)

        if file_dialog.exec():
            self.selected_files = file_dialog.selectedFiles()
            logging.debug(self.selected_files)

    def open_directory_dialog(self):
        """Opens directory selection window"""
        return QFileDialog.getExistingDirectory(self, "Select Folder of Photos")

    def load_folder_photos(self):
        logging.debug("Action: load photo directory")
        folder_path = self.open_directory_dialog()
        if folder_path:
            print(folder_path)
            # TODO: photo loading, separate logic
            # Perform operations to load and display photos from the folder
            # Example:
            # Iterate through photos in the folder and display them in the label
            # Here, you would use methods to load and display the photos in the label

            # For example, assuming photos are loaded into a list of paths:
            # photos = [list of photo file paths]
            # # Load the first photo from the folder
            # if photos:
            #     pixmap = QPixmap(photos[0])
            #     self.label.setPixmap(pixmap)

    def load_video_file(self):
        print("load_video_file")

    def set_fd_model(self, face_detector_name: str):
        self.th.set_fd_model(face_detector_name)

    def set_ad_model(self, text):
        self.th.set_ad_file(text)

    def btn_toggle_clicked(self):
        if self.btn_toggle.isChecked():
            self.btn_toggle.setText("Stop")
            self.start()
        else:
            self.btn_toggle.setText("Start")
            self.kill_thread()

    @pyqtSlot(QImage)
    def setImage(self, image):
        if self.live:
            pixmap = QPixmap.fromImage(image)
            self.scene.clear()
            self.scene.addPixmap(pixmap)
            self.view.setScene(self.scene)
            self.view.fitInView(self.scene.sceneRect(),
                                Qt.AspectRatioMode.KeepAspectRatio)

    @pyqtSlot()
    def removeImage(self):
        self.scene.clear()

    @pyqtSlot()
    def kill_thread(self):
        logging.debug("Finishing live recording...")
        cv2.destroyAllWindows()
        self.th.camera.release()
        self.live = False
        self.th.terminate()
        time.sleep(1)  # Give time for the thread to finish

    @pyqtSlot()
    def start(self):
        logging.debug("Starting live video capture...")
        self.live = True
        self.th.set_fd_model(self.fd_combobox.currentText())
        self.th.set_ad_file(self.ad_combobox.currentText())
        self.th.start()


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
