import sys

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from scipy.spatial.distance import cosine
import numpy as np
import imutils
import time
import cv2
import os

from ui.mainwindow_ui import Ui_MainWindow


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # prototxtPath = r"face_detector\deploy.prototxt"
        # weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        prototxtPath = os.path.join(
            os.getcwd(), 'face_detector', 'deploy.prototxt')
        weightsPath = os.path.join(
            os.getcwd(), 'face_detector', 'res10_300x300_ssd_iter_140000.caffemodel')

        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        maskNet = load_model("model.model")

        vs = VideoStream(src=0).start()
        while self._run_flag:
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "Without mask"
                color = (0, 255, 0) if label == "With mask" else (0, 0, 255)

                label = "{}: {:.2f}%".format(
                    label, max(mask, withoutMask) * 100)

                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv_img = frame
                self.change_pixmap_signal.emit(cv_img)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class FRaVGApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле mainWindow_ui.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.actionQuit.triggered.connect(QtWidgets.qApp.quit)
        # self.actionOpen.triggered.connect(browse_file)

        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def browse_folder(self):
        self.listWidget             # temp item
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose directory")  # temp item

        if directory:                                           # temp item
            for file_name in os.listdir(directory):
                self.listWidget.addItem(file_name)

    def browse_file(self):
        self.fPath
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выберите файл", '/home')[0]
        self.fPath.setText(fname)
        self.photoView.load(str(fname))
        self.photoView.scaled(150, 200)
        self.photoLbl.setPixmap(self.photoView)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.imgLabel.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1800, 700, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FRaVGApp()

    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
