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
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QRectF
from PyQt6.QtGui import QAction, QImage, QKeySequence, QPixmap, QIcon
from PyQt6.QtWidgets import (QApplication, QComboBox, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget, QFileDialog,
                             QGraphicsView, QGraphicsScene, QSplitter,
                             QGraphicsPixmapItem)

from config import Config
from face_detection import DetectedFace, FaceDetectors
from gui.processing import ProcessingThread
from gui.about import AboutDialog


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
        load_images_action = QAction("Select Images", self)
        load_images_action.triggered.connect(self.openFileDialogAndChooseImages)
        self.menu_file.addAction(load_images_action)
        # Scenario 3B
        load_images_from_dir_action = QAction("Select Images from Folder", self)
        # load_images_from_dir_action.triggered.connect()
        self.menu_file.addAction(load_images_from_dir_action)
        load_images_from_dir_action.triggered.connect(self.load_folder_images)
        # Scenario 2
        load_video_action = QAction("Select Video", self)
        load_video_action.triggered.connect(self.load_video_file)
        self.menu_file.addAction(load_video_action)
        exit = QAction("Exit", self, triggered=QApplication.quit)
        self.menu_file.addAction(exit)
        self.menu_about = self.menu.addMenu("&About")
        about = QAction("About", self)
        about.triggered.connect(self.show_about_page)
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
        self.image_label = QLabel("Source Filename")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.view = QGraphicsView(self)
        self.view.setMinimumSize(QSize(640, 480))
        self.view.setStyleSheet("border: 2px solid black;background-color: #333333;")
        self.scene = QGraphicsScene()
        if Config.WELCOME_IMAGE is not None:
            image_path = os.path.join(Config.STATIC_DIR_PATH,
                                      Config.WELCOME_IMAGE)
            pixmap = QPixmap(image_path)
            self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)
        self.view.fitInView(self.scene.sceneRect(),
                            Qt.AspectRatioMode.KeepAspectRatio)

        # Right layout only for QGraphics
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)

        # Nav. buttons
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prev_button = QPushButton("<")
        # self.prev_button.setStyleSheet("background: transparent; border: none; color: white; font-size: 18px;")
        self.next_button = QPushButton(">")
        # self.next_button.setStyleSheet("background: transparent; border: none; color: white; font-size: 18px;")
        self.buttons_layout.addWidget(self.prev_button)
        self.buttons_layout.addWidget(self.next_button)

        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)

        self.images = []
        self.total_images = len(self.images)
        self.current_image_index = 0

        # Right Panel
        right_layout.addWidget(self.view)
        right_layout.addLayout(self.buttons_layout)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(options_group_box)
        # splitter.addWidget(QWidget())
        right_panel_widget = QWidget()
        right_panel_widget.setLayout(right_layout)
        splitter.addWidget(right_panel_widget)
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
            self.images = file_dialog.selectedFiles()
            self.total_images = len(self.images)
            self.show_image(0)
            logging.debug(self.images)

    def open_directory_dialog(self):
        """Opens directory selection window"""
        return QFileDialog.getExistingDirectory(self, "Select Folder of images")

    def load_folder_images(self):
        logging.debug("Action: load photo directory")
        folder_path = self.open_directory_dialog()
        if folder_path:
            image_extensions = ['.jpg', '.jpeg', '.png']
            image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
                           if os.path.isfile(os.path.join(folder_path, file)) and
                           os.path.splitext(file)[1].lower() in image_extensions]

            self.images = image_files
            self.total_images = len(self.images)
            self.show_image(0)

    def show_about_page(self):
        about_dialog = AboutDialog(self)
        about_dialog.exec()

    def show_image(self, index):
        image_fullpath = self.images[index]
        image_file = os.path.basename(image_fullpath)
        pixmap = QPixmap(image_fullpath)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)
        self.view.fitInView(self.scene.sceneRect(),
                            Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setText(f"{image_file} ({index + 1} of {self.total_images})")

    def show_previous(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index)

    def show_next(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.show_image(self.current_image_index)

    def load_video_file(self):
        print("load_video_file")

    def set_fd_model(self, face_detector_name: str):
        self.th.set_fd_model(face_detector_name)

    def set_ad_model(self, text):
        self.th.set_ad_file(text)

    def btn_toggle_clicked(self):
        self.removeImage()
        if self.btn_toggle.isChecked():
            self.images = []
            self.total_images = 0
            self.btn_toggle.setText("Stop")
            self.prev_button.setHidden(True)
            self.next_button.setHidden(True)
            self.start()
        else:
            self.btn_toggle.setText("Start")
            self.prev_button.setHidden(False)
            self.next_button.setHidden(False)
            self.kill_thread()

    @pyqtSlot(QImage)
    def setImage(self, image):
        if self.live:
            pixmap = QPixmap.fromImage(image)
            self.scene.clear()
            self.scene.addPixmap(pixmap)
            # Set the scene rectangle to match the image size
            item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(item)
            self.scene.setSceneRect(QRectF(item.pixmap().rect()))
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
        self.image_label.setText("Live Video 🔴")
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
