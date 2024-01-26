import os
import sys
import time
from functools import reduce

import cv2
from PyQt6.QtCore import (Qt, QThread, pyqtSignal, pyqtSlot,
                          QSize, QRectF, QTimer, QUrl)
from PyQt6.QtGui import (QAction, QImage, QPixmap, QIcon, QKeyEvent, QPainter)
from PyQt6.QtWidgets import (QApplication, QComboBox, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget, QFileDialog,
                             QGraphicsView, QGraphicsScene, QSplitter,
                             QGraphicsPixmapItem, QSpacerItem, QProgressBar, QMessageBox)
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtMultimedia import QMediaPlayer

from config import Config
from face_detection import DetectedFace, FaceDetectors
from gui.processing import ProcessingThread, ImageProcessingThread, VideoProcessingThread
from gui.about import AboutDialog
from gui.video_preview import VideoPlayerWindow

"""
This file contains the core elements of the PyQt6 GUI application
for face detection and age prediction.
It is based on 3 different modes (scenarios):
1. Live Capture Mode - live video capture from webcam
2. Video Mode - video file processing
3. Picture Mode - image(s) processing
Also referred to as mode 1, mode 2, mode 3 respectively.
"""


HINTS = {
    "start": "Start by selecting source from Start menu above",
    "live": "You can switch the models and the changes will be reflected immediately on the video.\nClick on 'Stop Live Capture' button to stop this mode.",
    "video": "Select the desired models below and click on 'Start' button to start processing the video.",
    "images": "The images were loaded. You can preview them by using '<' and '>' buttons.\nSelect the desired models below and click on 'Start' button to start processing the images.",
    "no_images": "No images were selected/loaded.\nStart by selecting source from Start menu above",
    "images_after": "The images were processed. You can view them in the right panel.\nTo start a new processing, select source from Start menu above.",
    "images_loading": "Loading images and predicting age. Please wait..."
}


class GraphicsViewWithZoom(QGraphicsView):
    """Updated QGraphicsView with zooming capabilities"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.initialScaleFactor = None

    def wheelEvent(self, event):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        oldPos = self.mapToScene(event.position().toPoint())

        if self.initialScaleFactor is None:
            self.initialScaleFactor = self.transform().m11()

        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            if self.transform().m11() > self.initialScaleFactor:
                zoomFactor = zoomOutFactor
            else:
                return

        self.scale(zoomFactor, zoomFactor)
        newPos = self.mapToScene(event.position().toPoint())
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())


class MainWindow(QMainWindow):
    """Main window of the application"""
    def __init__(self):
        super().__init__()
        self.set_window_defaults()
        self.create_menu_bar()
        self.create_whole_window()
        self.create_processing_threads()

    def set_window_defaults(self):
        """Sets window title, size, icon and several
        variables needed for UI updates"""
        self.setWindowTitle(Config.TITLE)
        self.setGeometry(0, 0, 800, 500)
        # https://www.flaticon.com/free-icons/age-group
        # Age group icons created by Freepik - Flaticon
        self.setWindowIcon(QIcon(os.path.join(Config.STATIC_DIR_PATH, "age_group.png")))
        # for mode 3
        self.images = []
        self.total_images = len(self.images)
        self.current_image_index = 0
        # for mode 1
        self.live = False

    def show_modal_dialog(self):
        """Shows a modal dialog with OK and Cancel buttons for mode 2"""
        # Create a modal dialog with OK and Cancel buttons
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Warning")
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setText("Video preview and display is not currently available.\n"
                       "For this mode you should firstly choose the models and then Click 'Load Video'\n\n"
                       "Currently chosen models:\n"
                       f"Face detection: {self.fd_combobox.currentText()}\n"
                       f"Age prediction: {self.ad_combobox.currentText()}\n\n"
                        "Do you want to continue?"
                        "\n\nNote: The result file will be available in output directory named after "
                        "the timestamp of the operation")
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

        # Show the dialog and get the result
        result = dialog.exec()

        # Process the result
        if result == QMessageBox.StandardButton.Ok:
            return True
        elif result == QMessageBox.StandardButton.Cancel:
            return False

    def create_whole_window(self):
        """Combines left and right panels into one window"""

        # Create left and right panels
        self.create_left_panel()
        self.create_right_panel()

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.left_panel_widget)
        splitter.addWidget(self.right_panel_widget)
        splitter.setStretchFactor(0, 1)

        # Central widget
        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_right_panel(self):
        """Creates right panel with graphics, and utility buttons"""

        # Source label
        self.source_label = QLabel()
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.source_label.setHidden(True)

        # Graphics View
        self.view = GraphicsViewWithZoom(self)
        self.view.setMinimumSize(QSize(640, 480))
        self.view.setStyleSheet("border: 2px solid black;background-color: #333333;")
        self.scene = QGraphicsScene()
        if Config.WELCOME_IMAGE != "":
            image_path = os.path.join(Config.STATIC_DIR_PATH,
                                      Config.WELCOME_IMAGE)
            pixmap = QPixmap(image_path)
            self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)
        self.view.fitInView(self.scene.sceneRect(),
                            Qt.AspectRatioMode.KeepAspectRatio)

        # Right layout only for QGraphics
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.source_label)

        # Gallery Navigation buttons layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prev_button = QPushButton()
        self.next_button = QPushButton()
        # https://www.flaticon.com/free-icon/arrow-left_748099?term=arrow%20left&page=1&position=1&page=1&position=1&related_id=748099&origin=search
        prev_icon = QIcon(os.path.join(Config.STATIC_DIR_PATH,
                                       "arrow_left.png"))
        next_icon = QIcon(os.path.join(Config.STATIC_DIR_PATH,
                                       "arrow_right.png"))
        self.prev_button.setIcon(prev_icon)
        self.next_button.setIcon(next_icon)
        self.prev_button.setIconSize(QSize(40, 40))  # Set the icon size
        self.next_button.setIconSize(QSize(40, 40))  # Set the icon size
        self.buttons_layout.addWidget(self.prev_button)
        self.buttons_layout.addWidget(self.next_button)
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)
        self.prev_button.setHidden(True)
        self.next_button.setHidden(True)

        # Button layout for stopping live capture
        self.buttons_layout2 = QHBoxLayout()
        self.buttons_layout2.setContentsMargins(0, 0, 0, 0)
        self.buttons_layout2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stop_live_button = QPushButton("Stop Live Capture")
        self.stop_live_button.setStyleSheet("font-size: 16px;")
        self.buttons_layout2.addWidget(self.stop_live_button)
        self.stop_live_button.clicked.connect(self.stop_live_capture)
        self.stop_live_button.setHidden(True)

        # Right Panel
        right_layout.addWidget(self.view)
        right_layout.addLayout(self.buttons_layout)
        right_layout.addLayout(self.buttons_layout2)

        # Video widget for mode 2
        self.video_widget_layout = QVBoxLayout()
        self.video_widget = QVideoWidget()
        self.video_widget_layout.addWidget(self.video_widget)

        # Right Panel
        self.right_panel_widget = QWidget()
        self.right_panel_widget.setLayout(right_layout)

    def create_left_panel(self):
        """Creates left panel with settings including model selection
        for all modes, start button for 3 and progress bar for mode 2,3"""

        # Label for tutorial
        self.label_layout = QHBoxLayout()
        self.hint_label = QLabel("Start by selecting source from Start menu above")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet("font-size: 14px; font-style: italic; color: #999999;")
        self.hint_label.setWordWrap(True)
        self.hint_label.setFixedWidth(280)
        self.label_layout.addWidget(self.hint_label)
        self.label_layout.addStretch(1)
        self.label_layout.setContentsMargins(0, 0, 0, 0)
        self.spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.label_layout.addItem(self.spacer)

        # Model selection groupboxes
        self.create_group_face_det()
        self.create_group_age_det()

        # Constant Layout
        options_layout = QVBoxLayout()
        options_layout.addLayout(self.label_layout)
        options_layout.addWidget(self.face_model_group)
        options_layout.addWidget(self.age_model_group)

        # Additional layout for temp buttons
        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setSizePolicy(QSizePolicy.Policy.Preferred,
                                     QSizePolicy.Policy.Preferred)
        self.start_button.setHidden(True)
        buttons_layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_image_processing)

        # Progress bar
        pbar_layout = QHBoxLayout()
        self.pbar = QProgressBar(self)
        self.pbar.setHidden(True)
        pbar_layout.addWidget(self.pbar)

        # Final layout
        options_and_buttons_layout = QVBoxLayout()
        options_and_buttons_layout.addLayout(options_layout)
        options_and_buttons_layout.addLayout(buttons_layout)
        options_and_buttons_layout.addLayout(pbar_layout)
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        options_and_buttons_layout.addItem(spacer)

        # Left Panel
        left_panel_widget = QGroupBox()
        left_panel_widget.setLayout(options_and_buttons_layout)
        left_panel_widget.setMaximumWidth(300)
        self.left_panel_widget = left_panel_widget

    def create_processing_threads(self):
        """Creates threads for image processing"""
        # Thread for Live Capture Mode
        self.live_thread = ProcessingThread(self)
        self.live_thread.finished.connect(self.remove_image)
        self.live_thread.updateFrame.connect(self.set_image)

        # Thread for Picture Mode
        self.image_thread = ImageProcessingThread(self.images)
        self.image_thread.imagesProcessed.connect(self.load_images_to_display)
        self.image_thread.progress.connect(self.pbar.setValue)
        self.image_thread.finished.connect(self.pbar.reset)
        self.image_thread.finished.connect(self.pbar.hide)

        # Thread for File Video Mode
        self.video_thread = VideoProcessingThread()
        self.video_thread.videoProcessed.connect(self.load_video)
        self.video_thread.progress.connect(self.pbar.setValue)
        self.video_thread.finished.connect(self.pbar.reset)
        self.video_thread.finished.connect(self.pbar.hide)

    def create_menu_bar(self):
        """Creates a menu bar inc. actions and their connections"""
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
        load_images_from_dir_action.triggered.connect(self.load_images_from_folder)
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

    def create_graphics_view(self):
        self.view = GraphicsViewWithZoom(self)
        self.view.setMinimumSize(QSize(640, 480))
        self.view.setStyleSheet("border: 2px solid black;background-color: #333333;")
        self.scene = QGraphicsScene()
        if Config.WELCOME_IMAGE != "":
            image_path = os.path.join(Config.STATIC_DIR_PATH,
                                      Config.WELCOME_IMAGE)
            pixmap = QPixmap(image_path)
            self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)
        self.view.fitInView(self.scene.sceneRect(),
                            Qt.AspectRatioMode.KeepAspectRatio)

    def create_group_face_det(self):
        # Face detection Model Group
        self.face_model_group = QGroupBox("Face detection model")
        self.face_model_group.setSizePolicy(QSizePolicy.Policy.Preferred,
                                            QSizePolicy.Policy.Expanding)
        model_layout = QHBoxLayout()
        self.fd_combobox = QComboBox()
        for fd_model_name in Config.FACE_DETECTION_MODELS.keys():
            self.fd_combobox.addItem(fd_model_name)
        # Connections
        self.fd_combobox.currentTextChanged.connect(self.set_fd_model)
        # Face detection model layout
        model_layout.addWidget(QLabel("Model:"), 10)
        model_layout.addWidget(self.fd_combobox, 90)
        self.face_model_group.setLayout(model_layout)

    def create_group_age_det(self):
        # Age detection Model Group
        self.age_model_group = QGroupBox("Age detection model")
        self.age_model_group.setSizePolicy(QSizePolicy.Policy.Preferred,
                                           QSizePolicy.Policy.Expanding)
        model_layout = QHBoxLayout()
        self.ad_combobox = QComboBox()
        for model in Config.AGE_DETECTION_MODELS.keys():
            self.ad_combobox.addItem(model)
        # Connections
        self.ad_combobox.currentTextChanged.connect(self.set_ad_model)
        # Age detection model layout
        model_layout.addWidget(QLabel("Model:"), 10)
        model_layout.addWidget(self.ad_combobox, 90)
        self.age_model_group.setLayout(model_layout)

    def resizeEvent(self, event):
        """Rescales the image if the window is resized"""
        super().resizeEvent(event)
        self.show_image(self.current_image_index)

    def load_video(self, output_dir):
        self.video_window = VideoPlayerWindow(output_dir)
        self.video_window.show()

    def load_images_to_display(self, image_dir_path: str):
        self.hint_label.setText(HINTS.get("images_after"))
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [os.path.join(image_dir_path, file) for file in os.listdir(image_dir_path)
                       if os.path.isfile(os.path.join(image_dir_path, file)) and
                       os.path.splitext(file)[1].lower() in image_extensions]
        # self.current_image_index = 0
        self.images = image_files
        self.total_images = len(self.images)
        self.show_image(self.current_image_index)

    def reset_images(self):
        self.images = []
        self.total_images = len(self.images)
        self.current_image_index = 0

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

    def load_images_from_selection(self):
        self.stop_live_capture()
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)

        if file_dialog.exec():
            self.images = file_dialog.selectedFiles()
            self.current_image_index = 0
            self.total_images = len(self.images)
            if self.total_images > 0:
                self.next_button.setHidden(False)
                self.next_button.setEnabled(True)
                self.prev_button.setHidden(False)
                self.prev_button.setEnabled(True)
                self.start_button.setHidden(False)
                self.start_button.setEnabled(True)
                self.fd_combobox.setEnabled(True)
                self.ad_combobox.setEnabled(True)
                self.hint_label.setText(HINTS.get("images"))
                self.show_image(0)
            else:
                self.hint_label.setText(HINTS.get("no_images"))

    def open_directory_dialog(self):
        """Opens directory selection window"""
        return QFileDialog.getExistingDirectory(self, "Select Folder of images")

    def load_images_from_folder(self):
        self.stop_live_capture()
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
                self.hint_label.setText(HINTS.get("images"))
                self.prev_button.setHidden(False)
                self.next_button.setHidden(False)
                self.start_button.setHidden(False)
                self.start_button.setEnabled(True)
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
        if self.total_images == 0:
            return
        if self.total_images == 1:
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
        else:
            if index == self.total_images - 1:
                self.next_button.setEnabled(False)
                self.prev_button.setEnabled(True)
            elif index == 0:
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(True)
            else:
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
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
        self.source_label.setText(f"{image_file} ({index + 1} of {self.total_images})")
        self.source_label.setHidden(False)

    def show_previous(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index)

    def show_next(self):
        if self.current_image_index < self.total_images - 1:
            self.current_image_index += 1
            self.show_image(self.current_image_index)

    def load_video_file(self):
        """Loads video file and sets up the UI before processing"""
        result = self.show_modal_dialog()
        if result:
            file_dialog = QFileDialog()
            filepath, _ = file_dialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
            if filepath:
                self.pbar.setHidden(False)
                self.video_thread.set_ad_file(self.ad_combobox.currentText())
                self.video_thread.set_fd_model(self.fd_combobox.currentText())
                self.video_thread.set_video_path(filepath)
                self.video_thread.start()
                self.hint_label.setText("Video is being processed. Please wait...")

    def set_fd_model(self, face_detector_name: str):
        self.live_thread.set_fd_model(face_detector_name)

    def set_ad_model(self, text):
        self.live_thread.set_ad_file(text)

    def start_live_capture(self):
        """Sets the proper UI and starts live video capture from webcam"""
        self.remove_image()
        self.images = []
        self.total_images = 0
        self.fd_combobox.setEnabled(True)
        self.ad_combobox.setEnabled(True)
        self.start_button.setHidden(True)
        self.prev_button.setHidden(True)
        self.next_button.setHidden(True)
        self.stop_live_button.setHidden(False)
        self.source_label.setText("Live Video 🔴")
        self.source_label.setHidden(False)
        self.hint_label.setText(HINTS.get("live"))
        self.live = True
        self.start_live_vide_thread()

    def stop_live_capture(self):
        "Stops live capture and resets UI"
        self.kill_live_video_thread()
        self.reset_graphics_display()
        self.stop_live_button.setHidden(True)
        self.source_label.setHidden(True)
        self.hint_label.setText(HINTS.get("start"))

    def start_image_processing(self):
        """Starts processing images"""
        self.image_thread.set_images_paths_list(self.images)
        self.image_thread.set_fd_model(self.fd_combobox.currentText())
        self.image_thread.set_ad_file(self.ad_combobox.currentText())
        self.pbar.setHidden(False)
        self.hint_label.setText(HINTS.get("images_loading"))
        self.image_thread.start()
        self.fd_combobox.setEnabled(False)
        self.ad_combobox.setEnabled(False)
        self.start_button.setHidden(True)

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
    def kill_live_video_thread(self):
        cv2.destroyAllWindows()
        if type(self.live_thread.camera) == cv2.VideoCapture:
            self.live_thread.camera.release()
        self.live = False
        self.live_thread.terminate()
        time.sleep(2)  # Give time for the thread to finish

    @pyqtSlot()
    def start_live_vide_thread(self):
        self.live_thread.set_fd_model(self.fd_combobox.currentText())
        self.live_thread.set_ad_file(self.ad_combobox.currentText())
        self.live_thread.start()


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
