import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel)
from PyQt6.QtCore import Qt
from config import Config


class AboutDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("About")
        self.setMinimumSize(400, 400)
        layout = QVBoxLayout()
        heading_label = QLabel('<p align="center" style="font-size:16px; word-wrap: break-word;"><b>Age-detect app</b></p>')
        layout.addWidget(heading_label)

        description_label = QLabel(
            """<p style="font-size:14px">An ML-based age detection app created for the <i>Introduction to Machine Learning</i> course at Warsaw University of Technology.
            For more information, visit project <a href="https://github.com/Bartosz7/age-detect-app"> repository.</a></p>
            """
        )
        description_label.setWordWrap(True)
        description_label.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        description_label.setOpenExternalLinks(True)
        layout.addWidget(description_label)

        # List of authors
        authors_label = QLabel(
            '''<p style="font-size:16px">
                <div style="text-align:center"> <b>Authors</b> <br>
                Weronika Plichta<br>
                Szymon Trochimiak<br>
                Filip Kucia<br>
                Michał Taczała<br>
                Bartosz Grabek
                </div>
               </p>'''
        )
        layout.addWidget(authors_label)

        # Image of authors
        img_path = os.path.join(Config.STATIC_DIR_PATH, "authors.jpeg")
        image_label = QLabel(f'<img src="{img_path}" width="400" height="300">')
        layout.addWidget(image_label)

        self.setLayout(layout)
