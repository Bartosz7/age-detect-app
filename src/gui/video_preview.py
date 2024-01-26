from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtCore import QUrl


class VideoPlayerWindow(QMainWindow):
    def __init__(self, video_filepath):
        super(VideoPlayerWindow, self).__init__()
        self.resize(800, 600)
        self.setWindowTitle("Preview")

        self.media_player = QMediaPlayer(self)
        self.video_widget = QVideoWidget(self)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.mediaStatusChanged.connect(self.handle_media_status_changed)

        layout = QVBoxLayout()
        layout.addWidget(self.video_widget)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Load and play the video file
        self.load_and_play_video(video_filepath)

    def load_and_play_video(self, video_filepath):
        self.media_player.setSource(QUrl.fromLocalFile(video_filepath))
        self.media_player.play()

    def handle_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.close()
