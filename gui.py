import os
import sys
import time
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QApplication, QHBoxLayout, QGridLayout, QMessageBox
from PyQt5.QtCore import QSize, pyqtSlot, QTimer
from PyQt5.QtGui import QIcon, QPixmap

import main


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(self.sizeHint()) 
        self.setWindowTitle("AI Project Demo") 

        self.setStyleSheet('QWidget {font-family: Helvetica; font-size: 12px}')

        layout1 = QHBoxLayout() # parameters on left, image frames on right
        layout2 = QGridLayout() # parameters
        layout3 = QHBoxLayout() # frame

        layout1.setContentsMargins(20,20,20,20)
        layout1.setSpacing(20)

        # add widgets
        self.agentLabel = QLabel(self)
        self.agentLabel.setText('Number of Agents:')
        self.agentLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout2.addWidget(self.agentLabel, 0, 0)

        self.agentLine = QLineEdit(self)
        layout2.addWidget(self.agentLine, 0, 1)

        self.widthLabel = QLabel(self)
        self.widthLabel.setText("Width:")
        self.widthLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout2.addWidget(self.widthLabel, 1, 0)

        self.widthLine = QLineEdit(self)
        layout2.addWidget(self.widthLine, 1, 1)

        self.heightLabel = QLabel(self)
        self.heightLabel.setText("Height:")
        self.heightLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout2.addWidget(self.heightLabel, 2, 0)

        self.heightLine = QLineEdit(self)
        layout2.addWidget(self.heightLine, 2, 1)

        self.pybutton = QPushButton('Generate', self)
        self.pybutton.clicked.connect(self.generateMethod)
        layout2.addWidget(self.pybutton, 3, 1)

        layout1.addLayout(layout2)

        self.imgLabel = QLabel(self)
        self.imgLabel.setFixedSize(self.frameGeometry().height(), self.frameGeometry().height())
        layout3.addWidget(self.imgLabel)

        layout1.addLayout(layout3)

        widget = QWidget()
        widget.setLayout(layout1)
        self.setCentralWidget(widget)    

        self.framenum = 0

        self.pretrained_path = "utils/pretrained/1agent-999trials-1.0-weights-2020-08-16 06_00_28.038965.pt"

    def generateMethod(self): # start timer when 'Generate' button is clicked
        print('No. of Agents:', self.agentLine.text())
        try:
            no_of_trains = int(self.agentLine.text())
            self.aspect_ratio = int(self.widthLine.text()) / int(self.heightLine.text())
        except ValueError: # user enters a non-integer value
            self.aspect_ratio = 1
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Please enter an integer value.")
            msg.setIcon(QMessageBox.Critical)
            returnValue = msg.exec()
            if returnValue == QMessageBox.Ok:
                self.agentLine.clear()
                self.widthLine.clear()
                self.heightLine.clear()
                return
        print("Generating frames ...")
        # USE PRETRAINED MODEL TO GENERATE FRAMES HERE
        # SAVE FRAMES INTO tmp/frames folder
        main.render_frames(int(self.agentLine.text()), int(self.widthLine.text()), int(self.heightLine.text()), self.pretrained_path)
        self.agentLine.setReadOnly(True)
        self.widthLine.setReadOnly(True)
        self.heightLine.setReadOnly(True)
        self.pybutton.setText("Reset")
        self.pybutton.clicked.connect(self.resetMethod)
        self.timer = QTimer()
        self.timer.timeout.connect(self.animateFrame)
        self.timer.start(100)
    
    def resetMethod(self): # reset parameters
        try:
            self.timer.stop()
        except AttributeError:
            pass
        self.imgLabel.clear()
        self.agentLine.setReadOnly(False)
        self.agentLine.clear()
        self.widthLine.setReadOnly(False)
        self.widthLine.clear()
        self.heightLine.setReadOnly(False)
        self.heightLine.clear()
        self.pybutton.setText("Generate")
        self.pybutton.clicked.connect(self.generateMethod)

    def animateFrame(self): # timer changes image every 100ms, and stops after 500 images
        dir = os.path.join('tmp', 'frames')
        dir_list = os.listdir(dir)
        if self.framenum == len(dir_list):
            self.timer.stop()
        image_path = "tmp\\frames\\flatland_frame_0{}.png".format(str(self.framenum).zfill(3))
        print(image_path)
        frame = QPixmap(image_path)
        if self.framenum == 0:
            self.imgLabel.setFixedSize(self.frameGeometry().height()*self.aspect_ratio, self.frameGeometry().height())
        print(frame.scaled(self.imgLabel.size(), QtCore.Qt.KeepAspectRatio))
        self.imgLabel.setPixmap(frame.scaled(self.imgLabel.size(), QtCore.Qt.KeepAspectRatio))
        self.framenum += 1


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())