import os
import sys
import time
import logging
from functools import reduce

import cv2
from PyQt6.QtCore import (Qt, QThread, pyqtSignal, pyqtSlot,
                          QSize, QRectF, QTimer)
from PyQt6.QtGui import QAction, QImage, QPixmap, QIcon, QKeyEvent, QPainter
from PyQt6.QtWidgets import (QApplication, QComboBox, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget, QFileDialog,
                             QGraphicsView, QGraphicsScene, QSplitter,
                             QGraphicsPixmapItem, QSpacerItem, QProgressBar)

from config import Config
from face_detection import DetectedFace, FaceDetectors
from gui.processing import ProcessingThread, ImageProcessingThread
from gui.about import AboutDialog


class GraphicsViewWithZoom(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.initialScaleFactor = None

    def wheelEvent(self, event):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        # Save the scene pos
        oldPos = self.mapToScene(event.position().toPoint())

        # Save the initial scale factor
        if self.initialScaleFactor is None:
            self.initialScaleFactor = self.transform().m11()

        # Zoom
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            if self.transform().m11() > self.initialScaleFactor:
                zoomFactor = zoomOutFactor
            else:
                return
        self.scale(zoomFactor, zoomFactor)

        # Get the new position
        newPos = self.mapToScene(event.position().toPoint())

        # Move scene to old position
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())


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
        # Scenario 1
        start_live_action = QAction("Start Live Capture", self)
        start_live_action.triggered.connect(self.start_live_capture)
        self.menu_file.addAction(start_live_action)
        # Scenario 3A
        load_images_action = QAction("Select Image(s)", self)
        load_images_action.triggered.connect(self.load_images_from_selection)
        self.menu_file.addAction(load_images_action)
        # Scenario 3B
        load_images_from_dir_action = QAction("Select Images from Folder", self)
        self.menu_file.addAction(load_images_from_dir_action)
        load_images_from_dir_action.triggered.connect(self.load_folder_images)
        # Scenario 2
        load_video_action = QAction("Select Video", self)
        load_video_action.triggered.connect(self.load_video_file)
        self.menu_file.addAction(load_video_action)
        # Other Actions
        exit = QAction("Exit", self, triggered=QApplication.quit)
        self.menu_file.addAction(exit)
        self.menu_about = self.menu.addMenu("&About")
        about = QAction("About", self)
        about.triggered.connect(self.show_about_page)
        license = QAction("License", self,
                          triggered=QApplication.aboutQt)
        self.menu_about.addAction(about)
        self.menu_about.addAction(license)

        # Left Panel: options layout
        # create a new QHBoxLayout() with single label
        self.label_layout = QHBoxLayout()
        # make label look nicer
        self.label_desc = QLabel("Start by selecting source from Start menu above")
        self.label_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_desc.setStyleSheet("font-style: italic; color: #999999;")
        self.label_desc.setWordWrap(True)
        self.label_desc.setFixedWidth(280)
        self.label_layout.addWidget(self.label_desc)
        self.label_layout.addStretch(1)
        self.label_layout.setContentsMargins(0, 0, 0, 0)
        self.spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.label_layout.addItem(self.spacer)
        self.create_group_face_det()
        self.create_group_age_det()
        options_layout = QVBoxLayout()
        options_layout.addLayout(self.label_layout)
        options_layout.addWidget(self.group_face_model)
        options_layout.addWidget(self.group_age_model)

        buttons_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setSizePolicy(QSizePolicy.Policy.Preferred,
                                      QSizePolicy.Policy.Preferred)
        self.start_btn.setHidden(True)
        buttons_layout.addWidget(self.start_btn)
        # add progress bar
        pbar_layout = QHBoxLayout()
        self.pbar = QProgressBar(self)
        self.pbar.setHidden(True)
        pbar_layout.addWidget(self.pbar)

        options_and_buttons_layout = QVBoxLayout()
        options_and_buttons_layout.addLayout(options_layout)
        options_and_buttons_layout.addLayout(buttons_layout)
        options_and_buttons_layout.addLayout(pbar_layout)
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        options_and_buttons_layout.addItem(spacer)

        options_group_box = QGroupBox("Settings")
        options_group_box.setLayout(options_and_buttons_layout)
        options_group_box.setMaximumWidth(300)

        # Right Panel: Graphics and navigation
        self.image_label = QLabel("Source Filename")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.view = GraphicsViewWithZoom(self)
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
        self.next_button = QPushButton(">")
        self.buttons_layout.addWidget(self.prev_button)
        self.buttons_layout.addWidget(self.next_button)
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)
        # hide at first
        self.prev_button.setHidden(True)
        self.next_button.setHidden(True)

        # additional button layour for stopping live video
        self.buttons_layout2 = QHBoxLayout()
        self.buttons_layout2.setContentsMargins(0, 0, 0, 0)
        self.buttons_layout2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stop_live_btn = QPushButton("Stop Live Capture")
        self.buttons_layout2.addWidget(self.stop_live_btn)
        self.stop_live_btn.clicked.connect(self.stop_live_capture)
        self.stop_live_btn.setHidden(True)

        self.images = []
        self.total_images = len(self.images)
        self.current_image_index = 0

        # Right Panel
        right_layout.addWidget(self.view)
        right_layout.addLayout(self.buttons_layout)
        right_layout.addLayout(self.buttons_layout2)

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
        self.th.finished.connect(self.remove_image)
        self.th.updateFrame.connect(self.set_image)
        # 2nd thread for image processing
        self.image_thread = ImageProcessingThread(self.images)
        self.image_thread.imagesProcessed.connect(self.load_images_to_display)
        self.image_thread.progress.connect(self.pbar.setValue)
        self.image_thread.finished.connect(self.pbar.reset)
        self.image_thread.finished.connect(self.pbar.hide)
        # Connections
        self.connect_all()

        # Video Player timer 
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)

    def load_images_to_display(self, image_dir_path: str):
        print(image_dir_path)
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [os.path.join(image_dir_path, file) for file in os.listdir(image_dir_path)
                       if os.path.isfile(os.path.join(image_dir_path, file)) and
                       os.path.splitext(file)[1].lower() in image_extensions]
        self.current_image_index = 0
        self.images = image_files
        self.total_images = len(self.images)
        self.prev_button.setHidden(False)
        self.next_button.setHidden(False)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(True)
        self.show_image(0)

    def reset_images(self):
        self.images = []
        self.total_images = len(self.images)
        self.current_image_index = 0

    def update_video(self):
        ret, frame = self.capture.read()

        if ret:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            self.scene.clear()
            self.scene.addPixmap(QPixmap.fromImage(q_image))
            self.view.fitInView(self.scene.sceneRect(),
                                Qt.AspectRatioMode.KeepAspectRatio)
            # current_frame = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            self.timer.stop()
            self.capture.release()

    def event(self, event):
        if event.type() == QKeyEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Right:
                self.next_button.click()
                return True
            elif event.key() == Qt.Key.Key_Left:
                self.prev_button.click()
                return True
            return False  # Indicate the event was handled
        return super().event(event)

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
        self.start_btn.clicked.connect(self.start_btn_clicked)
        self.fd_combobox.currentTextChanged.connect(self.set_fd_model)
        self.ad_combobox.currentTextChanged.connect(self.set_ad_model)

    def load_images_from_selection(self):
        self.stop_live_capture()
        self.timer.stop()
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)

        if file_dialog.exec():
            self.images = file_dialog.selectedFiles()
            self.current_image_index = 0
            self.total_images = len(self.images)
            if self.total_images > 0:
                if self.total_images > 1:
                    self.label_desc.setText("The images were loaded. You can preview them by using '<' and '>' buttons. Select the desired models below and click on 'Start' button to start processing the images")
                    self.prev_button.setHidden(False)
                    self.next_button.setHidden(False)
                    self.prev_button.setEnabled(False)
                    self.next_button.setEnabled(True)
                    self.start_btn.setHidden(False)
                    self.start_btn.setEnabled(True)
                if self.total_images == 1:
                    self.label_desc.setText("The images were loaded. You can preview them by using '<' and '>' buttons. Select the desired models below and click on 'Start' button to start processing the images")
                    self.prev_button.setHidden(True)
                    self.next_button.setHidden(True)
                    self.start_btn.setHidden(False)
                    self.start_btn.setEnabled(True)
                self.show_image(0)

    def open_directory_dialog(self):
        """Opens directory selection window"""
        return QFileDialog.getExistingDirectory(self, "Select Folder of images")

    def load_folder_images(self):
        logging.debug("Action: load photo directory")
        self.timer.stop()
        folder_path = self.open_directory_dialog()
        if folder_path:
            image_extensions = ['.jpg', '.jpeg', '.png']
            image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
                           if os.path.isfile(os.path.join(folder_path, file)) and
                           os.path.splitext(file)[1].lower() in image_extensions]
            if len(image_files) > 0:
                self.current_image_index = 0
                self.images = image_files
                self.total_images = len(self.images)
                if self.total_images > 1:
                    self.label_desc.setText("The images were loaded. You can preview them by using '<' and '>' buttons. Select the desired models below and click on 'Start' button to start processing the images")
                    self.prev_button.setHidden(False)
                    self.next_button.setHidden(False)
                    self.prev_button.setEnabled(False)
                    self.next_button.setEnabled(True)
                    self.start_btn.setHidden(False)
                    self.start_btn.setEnabled(True)
                if self.total_images == 1:
                    self.label_desc.setText("The images were loaded. You can preview them by using '<' and '>' buttons. Select the desired models below and click on 'Start' button to start processing the images")
                    self.prev_button.setHidden(True)
                    self.next_button.setHidden(True)
                    self.start_btn.setHidden(False)
                    self.start_btn.setEnabled(True)
                self.show_image(0)

    def reset_graphics_display(self):
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.view.fitInView(self.scene.sceneRect(),
                            Qt.AspectRatioMode.KeepAspectRatio)

    def show_about_page(self):
        about_dialog = AboutDialog(self)
        about_dialog.exec()

    def show_image(self, index):
        image_fullpath = self.images[index]
        image_file = os.path.basename(image_fullpath)
        pixmap = QPixmap(image_fullpath)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        rectF = QRectF(pixmap.rect())
        self.scene.setSceneRect(rectF)
        self.view.setScene(self.scene)
        self.view.fitInView(self.scene.sceneRect(),
                            Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setText(f"{image_file} ({index + 1} of {self.total_images})")

    def show_previous(self):
        self.next_button.setEnabled(True)
        if self.current_image_index == 1:
            self.prev_button.setEnabled(False)
            self.current_image_index -= 1
            self.show_image(self.current_image_index)
        if self.current_image_index > 1:
            self.prev_button.setEnabled(True)
            self.current_image_index -= 1
            self.show_image(self.current_image_index)

    def show_next(self):
        self.prev_button.setEnabled(True)
        if self.current_image_index == len(self.images) - 2:
            self.next_button.setEnabled(False)
            self.current_image_index += 1
            self.show_image(self.current_image_index)
        if self.current_image_index < len(self.images) - 2:
            self.next_button.setEnabled(True)
            self.current_image_index += 1
            self.show_image(self.current_image_index)

    def load_video_file(self):
        self.timer.stop()
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")

        if filepath:
            self.capture = cv2.VideoCapture(filepath)
            self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            # self.slider.setMinimum(0)
            # self.slider.setMaximum(self.total_frames - 1)
            self.play_video()

    def play_video(self):
        self.reset_graphics_display()
        if not self.timer.isActive():
            self.timer.start(10)  # Adjust timer interval as needed (33 milliseconds for ~30 fps)

    def set_fd_model(self, face_detector_name: str):
        self.th.set_fd_model(face_detector_name)

    def set_ad_model(self, text):
        self.th.set_ad_file(text)

    def start_img_scenario(self):
        """Loads images, sets up the UI before processing"""
        pass

    def start_img_processing(self):
        """Starts loaded images processing"""
        pass

    def start_live_capture(self):
        """Sets the proper UI and starts live video capture from webcam"""
        self.remove_image()
        self.timer.stop()
        self.images = []
        self.total_images = 0
        self.start_btn.setHidden(True)
        self.prev_button.setHidden(True)
        self.next_button.setHidden(True)
        self.stop_live_btn.setHidden(False)
        self.image_label.setText("Live Video 🔴")
        self.label_desc.setText("You can switch the models and the changes will be reflected immediately on the video.\nClick on 'Stop Live Capture' button to stop this mode.")
        self.live = True
        self.start_thread()

    def stop_live_capture(self):
        "Stops live capture and resets UI"
        self.kill_thread()
        self.reset_graphics_display()
        self.stop_live_btn.setHidden(True)
        self.image_label.setText("Source")
        self.label_desc.setText("Start by selecting source from Start menu above")

    def start_btn_clicked(self):
        """Starts processing images"""
        self.image_thread.set_images_paths_list(self.images)
        self.image_thread.set_fd_model(self.fd_combobox.currentText())
        self.image_thread.set_ad_file(self.ad_combobox.currentText())
        self.pbar.setHidden(False)
        self.image_thread.start()
        self.start_btn.setEnabled(False)
        # self.remove_image()
        # if self.start_btn.isChecked():
        #     self.timer.stop()
        #     self.images = []
        #     self.total_images = 0
        #     self.start_btn.setText("Stop")
        #     self.prev_button.setHidden(True)
        #     self.next_button.setHidden(True)
        #     self.start_thread()
        # else:
        #     self.start_btn.setText("Start")
        #     self.kill_thread()
        #     self.reset_graphics_display()

    @pyqtSlot(QImage)
    def set_image(self, image):
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
    def remove_image(self):
        self.scene.clear()

    @pyqtSlot()
    def kill_thread(self):
        logging.debug("Finishing live recording...")
        cv2.destroyAllWindows()  # TODO: check if needed
        if type(self.th.camera) == cv2.VideoCapture:
            self.th.camera.release()
        self.live = False
        self.th.terminate()
        time.sleep(2)  # Give time for the thread to finish

    @pyqtSlot()
    def start_thread(self):
        logging.debug("Starting live video capture...")
        self.th.set_fd_model(self.fd_combobox.currentText())
        self.th.set_ad_file(self.ad_combobox.currentText())
        self.th.start()


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
