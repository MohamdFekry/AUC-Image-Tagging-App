import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QPushButton,
    QInputDialog,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5 import QtCore

import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from googletrans import Translator
import pandas as pd
import os
import ntpath


class ImageScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel(self)
        self.setWidget(self.label)
        self.setWidgetResizable(True)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def set_image(self, pixmap):
        self.label.setPixmap(pixmap)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.adjustSize()


class ImageAndTextDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("AUC Image Tagging")
        self.setWindowIcon(QIcon("icon.ico"))
        self.setGeometry(100, 100, 800, 600)
        # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

        self.on_start()

        # Create widgets
        self.image_scroll_area = ImageScrollArea(self)
        self.image_scroll_area.setMinimumSize(self.width() // 3, self.height() // 3)
        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.open_image_dialog)

        self.logo_label = QLabel(self)
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
        logo_pixmap = QPixmap("pic.png")
        self.logo_label.setPixmap(logo_pixmap.scaled(200, 200))  # Adjust logo size here

        # Layout
        main_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        left_layout.addWidget(self.upload_button)
        left_layout.addWidget(self.image_scroll_area, alignment=QtCore.Qt.AlignCenter)

        # Manual control of the logo position
        left_layout.addStretch(1)
        left_layout.addWidget(self.logo_label, alignment=QtCore.Qt.AlignCenter)
        left_layout.addStretch(
            2
        )  # Adjust the number of stretches to control the position

        self.mantagbutton = QPushButton("Manual Tags", self)
        self.mantagbutton.setFont(QFont("Georgia", 15))
        # self.mantagbutton.setGeometry(700, 150, 100, 40)
        self.mantagbutton.clicked.connect(self.mantag_clicked)

        self.storebutton = QPushButton("Store", self)
        # self.storebutton.raise_()
        # self.storebutton.setGeometry(700, 150, 100, 40)
        self.storebutton.setFont(QFont("Georgia", 23))
        self.storebutton.clicked.connect(self.store_clicked)

        self.infobutton = QPushButton("Help", self)
        self.infobutton.setFont(QFont("Georgia", 15))
        self.infobutton.setStyleSheet(
            "QPushButton"
            "{"
            "border : 3px solid red;"
            "background : lightblue;"
            "}"
            "QPushButton:hover"
            "{"
            "border : 3px solid purple;"
            "background : orange;"
            "}"
        )
        self.infobutton.clicked.connect(self.info_clicked)

        # creating a QListWidget
        self.list_widget = QListWidget(self)
        # list widget items
        for i in range(40):
            # self.list_widget.addItem(QListWidgetItem(f"Text {i + 1}"))
            it = QListWidgetItem(f"Text {i + 1}")
            it.setFont(QFont("Times New Roman", 19))
            self.list_widget.addItem(it)

        self.list_widget.setStyleSheet(
            "QListWidget"
            "{"
            "border : 2px solid black;"
            "background : lightgreen;"
            "}"
            "QListWidget::item"
            "{"
            "border : 2px solid black;"
            "}"
            "QListWidget QScrollBar"
            "{"
            "background : lightblue;"
            "}"
            "QListView::item:hover"
            "{"
            "border : 2px solid black;"
            "background : green;"
            "}"
            "QListView::item:selected"
            "{"
            "border : 2px solid black;"
            "background : green;"
            "}"
        )

        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)

        # for text_label in self.text_labels:
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.list_widget)
        scroll_area.setWidgetResizable(True)
        right_layout.addWidget(scroll_area)

        scroll_area = QScrollArea(self)
        scroll_widget = QWidget(self)
        scroll_widget.setLayout(right_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)

        main_layout.addLayout(left_layout)
        main_layout.addWidget(scroll_area)

        # Center the image on the left side
        left_layout.setAlignment(QtCore.Qt.AlignTop)

        # Store box properties
        self.box_width = self.width() // 3
        self.box_height = self.height() // 3

    def on_start(self):
        self.file_name = ""

        self.incep_model = timm.create_model("inception_v4", pretrained=True)
        self.incep_model.eval()
        incep_config = resolve_data_config({}, model=self.incep_model)
        self.incep_transform = create_transform(**incep_config)

        self.res_model = timm.create_model("cspresnet50", pretrained=True)
        self.res_model.eval()
        res_config = resolve_data_config({}, model=self.res_model)
        self.res_transform = create_transform(**res_config)
        with open("imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]

    def resizeEvent(self, event):
        self.storebutton.setGeometry(
            self.rect().width() - 141,
            self.rect().height() - 73,
            100,
            50,
        )
        self.mantagbutton.setGeometry(
            120,
            self.rect().height() - 73,
            150,
            50,
        )
        self.infobutton.setGeometry(
            20,
            self.rect().height() - 73,
            75,
            50,
        )
        QWidget.resizeEvent(self, event)

    def path_leaf(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    def ML(self, img):
        incep_tensor = self.incep_transform(img).unsqueeze(0)

        with torch.no_grad():
            out = self.incep_model(incep_tensor)
        incep_probabilities = torch.nn.functional.softmax(out[0], dim=0)

        res_tensor = self.res_transform(img).unsqueeze(0)

        with torch.no_grad():
            out = self.res_model(res_tensor)
        res_probabilities = torch.nn.functional.softmax(out[0], dim=0)

        res_top15_prob, res_top15_catid = torch.topk(res_probabilities, 15)
        incep_top15_prob, incep_top15_catid = torch.topk(incep_probabilities, 15)
        self.categs = [self.categories[x] for x in res_top15_catid] + [
            self.categories[y] for y in incep_top15_catid
        ]
        self.categs = list(dict.fromkeys(self.categs))

    def open_image_dialog(self):
        self.file_name = ""
        options = QFileDialog.Options()
        self.file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Images (*.png *.jpg *.bmp);;All Files (*)",
            options=options,
        )

        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setSelected(False)

        if self.file_name:
            pixmap = QPixmap(self.file_name)

            max_width = self.width() // 2
            max_height = self.height() // 2

            # Scale the image if it exceeds half of the page's size
            if pixmap.width() > max_width or pixmap.height() > max_height:
                scaled_pixmap = pixmap.scaled(
                    max_width, max_height, QtCore.Qt.KeepAspectRatio
                )
                self.image_scroll_area.set_image(scaled_pixmap)
                self.image_scroll_area.setFixedSize(max_width, max_height)

            else:
                self.image_scroll_area.set_image(pixmap)
                self.image_scroll_area.setFixedSize(pixmap.size())

            # Store current image box properties
            self.box_width, self.box_height = (
                self.image_scroll_area.width(),
                self.image_scroll_area.height(),
            )

            self.path = self.file_name
            img = Image.open(self.path).convert("RGB")
            self.ML(img)

            stopIndex = len(self.categs)
            for i in range(stopIndex):
                if self.list_widget.item(i):
                    self.list_widget.item(i).setText(self.categs[i])
                else:
                    it = QListWidgetItem(self.categs[i])
                    it.setFont(QFont("Times New Roman", 19))
                    self.list_widget.addItem(it)

            for i in range(stopIndex, 40):
                if self.list_widget.item(stopIndex):
                    self.list_widget.takeItem(stopIndex)

        else:
            self.image_scroll_area.set_image(QPixmap())
            self.image_scroll_area.setFixedSize(self.box_width, self.box_height)

    def mantag_clicked(self):
        self.tag_list = []
        if self.file_name:
            self.tag_list = []
            self.userTag, stOK = QInputDialog(
                None, QtCore.Qt.WindowCloseButtonHint
            ).getText(
                self, "Input Tag", "Enter your tags seperated by comma or one tag:"
            )

            if stOK and self.file_name:
                self.tag_list = self.userTag.split(",")
                for usrtag in self.tag_list:
                    it = QListWidgetItem(usrtag)
                    it.setFont(QFont("Times New Roman", 19))
                    self.list_widget.addItem(it)
                    it.setSelected(True)

    def info_clicked(self):
        # norm = "\033[0m"
        # BOLD = "\033[1m"
        # UNDERLINE = "\033[4m"
        # PURPLE = "\033[95m"
        # CYAN = "\033[96m"
        # DARKCYAN = "\033[36m"
        # BLUE = "\033[94m"
        # GREEN = "\033[92m"
        # YELLOW = "\033[93m"
        # RED = "\033[91m"

        message = """<!DOCTYPE html>
                <html>
                <head>
                <style>
                  /* Add your CSS styles here */
                  .red-heading {
                    color: red;
                    font-size: 20px;
                  }
                  .larger-text {
                    font-size: 15px;
                  }
                  .larger-list {
                    list-style-type: decimal;
                    padding-left: 1em;
                  }

                </style>
                </head>
                <body>
                  <h2 class="red-heading">How To Use The App:</h2>
                  <ol class="larger-list larger-text">
                    <li>Click on the "<strong>Upload Image</strong>" button on the screen's top left.</li>
                    <li>Choose the photo from your PC and click okay.</li>
                    <li>Tags are automatically generated on the right, and to select a tag, just click on it, and you can unselect the tag by clicking on it once more.</li>
                    <li>Choose your tags from the generated tags on the right.</li>
                    <li>If you want to add tags that are not available, click on the "<strong>Manual Tags</strong>" button at the bottom left and follow the instructions.</li>
                    <li>Manual tags will be added to the list and automatically selected. They can be unselected too.</li>
                    <li>Press the "<strong>Store</strong>" button and wait until your image is uploaded. Once the image is uploaded, the tags list will be responsive to the mouse's hovering over it.</li>
                    <li>Finally, you will find the image and the list of tags in Arabic and English uploaded to the following drive link: <a href="https://drive.google.com/drive/folders/1VxcpHDDE7Psdyvfm_tcBBQ_RY2D6Snf2"><strong>https://drive.google.com/drive/folders/1VxcpHDDE7Psdyvfm_tcBBQ_RY2D6Snf2</strong></a></li>
                  </ol>
                  <h2 class="red-heading">Contact:</h2>
                  <p class="larger-text">Thank you for using the AUC Image Tagging APP. If you need any assistance, please don't hesitate to contact us, the developers Ahmad Mohamed and Mohamed Fekry, on the following emails:</p>
                  <ul class="larger-text">
                    <li><strong>Email:</strong> <a href="mailto:kongo@aucegypt.edu"><strong>kongo@aucegypt.edu</strong></a></li>
                    <li><strong>Email:</strong> <a href="mailto:mfeldahby@aucegypt.edu"><strong>mfeldahby@aucegypt.edu</strong></a></li>
                  </ul>
                </body>
                </html>
                """

        msg = QMessageBox(QMessageBox.Information, "HELP", message, QMessageBox.Ok)
        # msg.setStyleSheet("QLabel {min-width: 300px; }")
        msg.exec()
        # msg.setIcon(QMessageBox.Information)

        # temp = (
        #    "write \nlkkkkkkkkkkk\n;lllllllkmkmlkjnnhk"
        #    + "\033[91m"
        #    + "sgdaggasdvcxb\nfrgararvnnhymmfhfghn\ndhndjkmdkhjndmnmagkflnhlnsthjsslntubsaybglasdndbnsidtngbbgyahnszubbhsksnjkbghistnghhby"
        # )
        # print(temp)
        # msg.setText(temp)
        # msg.exec_()

    def store_clicked(self):
        checked = []
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            if it.isSelected():
                checked.append(it.text())

        try:
            os.remove("Images & Tags.xlsx")
        except:
            pass

        if self.file_name and checked:
            # print(checked)

            gauth = GoogleAuth()
            gauth.LoadCredentialsFile("credentials.json")
            if gauth.credentials is None:
                # Authenticate if they're not there
                gauth.GetFlow()
                gauth.flow.params.update({"access_type": "offline"})
                gauth.flow.params.update({"approval_prompt": "force"})
                gauth.LocalWebserverAuth()
                # gauth.CommandLineAuth()

            elif gauth.access_token_expired:
                # Refresh them if expired
                gauth.Refresh()

            else:
                # Initialize the saved creds
                gauth.Authorize()
            # Save the current credentials to a file
            gauth.SaveCredentialsFile("credentials.json")

            drive = GoogleDrive(gauth)

            file_list = drive.ListFile(
                {
                    "q": "'{}' in parents and trashed=false".format(
                        "1VxcpHDDE7Psdyvfm_tcBBQ_RY2D6Snf2"
                    )
                }
            ).GetList()
            for file in file_list:
                if file["title"] == "Images & Tags.xlsx":
                    file.GetContentFile(file["title"])
                    file.Trash()
                    df = pd.read_excel("Images & Tags.xlsx")
                    os.remove("Images & Tags.xlsx")
                    break
            else:
                df = pd.DataFrame(columns=["ImageName", "Tags", "علامات"])

            engtags = ""
            for tag in checked:
                engtags += tag
                engtags += ", "
            engtags = engtags[:-2]
            translator = Translator()
            aratags = translator.translate(engtags, src="en", dest="ar")

            path2save = self.path_leaf(self.path)

            updating = False

            if path2save in df["ImageName"].values:
                df.loc[df["ImageName"] == path2save, "Tags"] = engtags
                df.loc[df["ImageName"] == path2save, "علامات"] = aratags.text
                updating = True
            else:
                new_row = pd.DataFrame(
                    {"ImageName": path2save, "Tags": engtags, "علامات": aratags.text},
                    index=[0],
                )
                df = pd.concat([new_row, df.loc[:]]).reset_index(drop=True)

            df.to_excel("Images & Tags.xlsx", index=False)

            gfile = drive.CreateFile(
                {
                    "parents": [{"id": "1VxcpHDDE7Psdyvfm_tcBBQ_RY2D6Snf2"}],
                    "title": "Images & Tags.xlsx",
                }
            )
            # Read file and set it as the content of this instance.
            gfile.SetContentFile("Images & Tags.xlsx")
            gfile.Upload()  # Upload the file.

            if not updating:
                gfile = drive.CreateFile(
                    {
                        "parents": [{"id": "1VxcpHDDE7Psdyvfm_tcBBQ_RY2D6Snf2"}],
                        "title": path2save,
                    }
                )
                # Read file and set it as the content of this instance.
                gfile.SetContentFile(self.path)
                gfile.Upload()  # Upload the file.

                os.remove("Images & Tags.xlsx")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageAndTextDisplay()

    window.show()
    window.storebutton.raise_()
    window.mantagbutton.raise_()
    window.infobutton.raise_()
    sys.exit(app.exec_())
