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

import mainwindow


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
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
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

                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv_img = frame
                self.change_pixmap_signal.emit(cv_img)


    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class FRaVGApp(QtWidgets.QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле mainWindow.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.Exit.triggered.connect(QtWidgets.qApp.quit)
        self.logoutAction.setDisabled(True)
        self.logoutAction.triggered.connect(self.logout_menus)
        self.authAction.triggered.connect(lambda: self.auth())
        self.UserRmv.triggered.connect(lambda: self.accounts('rmUser'))
        self.RemoveGuest.triggered.connect(lambda: self.accounts('rmGuest'))
        self.GuestAdd.triggered.connect(lambda: self.accounts('addGuest'))
        self.ModifyGuest.triggered.connect(lambda: self.accounts('modGuest'))
        self.UserModify.triggered.connect(lambda: self.accounts('modUser'))

        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def unlock_menus(self, role):           # role - тип учетной записи (1 - Админ, 2 - обычный пользователь)
        if role == 1:
            self.GuestMenu.setEnabled(True)
            self.authAction.setDisabled(True)
            self.logoutAction.setEnabled(True)
        elif role == 2:
            self.GuestMenu.setEnabled(True)
            self.UsersMenu.setEnabled(True)
            self.authAction.setDisabled(True)
            self.logoutAction.setEnabled(True)

    def logout_menus(self):
        self.GuestMenu.setEnabled(False)
        self.UsersMenu.setEnabled(False)
        self.authAction.setEnabled(True)
        self.logoutAction.setEnabled(False)


    def auth(self):
        self.dialog = QtWidgets.QDialog(self)
        self.dialog.setWindowTitle('Авторизация')
        self.dialog.setModal(True)
        fbox = QtWidgets.QFormLayout()
        loginField = QtWidgets.QLineEdit()
        passField = QtWidgets.QLineEdit()
        passField.setEchoMode(QtWidgets.QLineEdit.Password)
        fbox.addRow('&Имя пользователя:', loginField)
        fbox.addRow('&Пароль:', passField)
        hbox = QtWidgets.QHBoxLayout()
        btnOK = QtWidgets.QPushButton('&OK')
        btnCancel = QtWidgets.QPushButton('&Отмена')
        hbox.addWidget(btnOK)
        hbox.addWidget(btnCancel)
        fbox.addRow(hbox)
        self.dialog.setLayout(fbox)
        btnCancel.clicked.connect(self.dialog.close)
        btnOK.clicked.connect(lambda: self.isValid(loginField.text(), passField.text()))
        self.dialog.adjustSize()
        self.dialog.show()
        self.dialog.exec_()

    def accounts(self, title):
        self.accWindow = QtWidgets.QDialog(self)
        self.accWindow.setModal(True)
        # self.prio = 1
        print(title)
        fbox = QtWidgets.QFormLayout()
        # self.photoView = QtWidgets.QGraphicsView()          # возможно надо удалить этот self вначале       self.photoView = QtWidgets.QGraphicsView() self.photoView = QtGui.QPixmap()
        # self.photoView.setContentsMargins(50, -1, 50, -1)
        self.photoLbl = QtWidgets.QLabel(self)
        self.photoView = QtGui.QPixmap("A_Fishful_of_Dollars.jpg")
        self.photoView.scaled(150, 200)
        self.photoLbl.setPixmap(self.photoView)
        fbox.addRow(self.photoLbl)
        self.fPath = QtWidgets.QLineEdit()
        self.fPath.setText('Укажите путь к файлу')
        self.fPath.setDragEnabled(True)
        self.btnBrowse = QtWidgets.QPushButton('&Обзор')
        self.btnDirBrowse = QtWidgets.QPushButton('&Фото')
        # btnBrowse.setFixedWidth(45)
        hPath = QtWidgets.QHBoxLayout()
        hPath.addWidget(self.fPath)
        hPath.addWidget(self.btnBrowse)
        hPath.addWidget(self.btnDirBrowse)
        fbox.addRow(hPath)
        ubox = QtWidgets.QFormLayout()
        loginField = QtWidgets.QLineEdit()
        passField = QtWidgets.QLineEdit()
        passField.setEchoMode(QtWidgets.QLineEdit.Password)
        extraBox = QtWidgets.QHBoxLayout()
        prioLbl = QtWidgets.QLineEdit('&Приоритет:')
        isAdmin = QtWidgets.QCheckBox('&Администратор', self)
        prioBox = QtWidgets.QComboBox()  # Приоритет приветствия
        prioBox.addItems(['1-обычный', '2-Высокий', '3-Первоочередной'])        # @TODO Исправить
        extraBox.addWidget(prioBox)
        if title == 'rmUser':
            self.accWindow.setWindowTitle('Удаление учетной записи пользователя')
        elif title == 'addGuest':
            self.accWindow.setWindowTitle('Добавить гостя')
            fbox.addRow(extraBox)
        elif title == 'addUser':
            self.accWindow.setWindowTitle('Создание учетной записи пользователя')
            ubox.addRow('&Имя пользователя:', loginField)
            ubox.addRow('&Пароль:', passField)
            fbox.addRow(ubox)
            extraBox.addWidget(isAdmin)
            fbox.addRow(extraBox)
        elif title == 'modGuest':
            self.accWindow.setWindowTitle('Изменение данных гостя')
            fbox.addRow(extraBox)
        elif title == 'modUser':
            self.accWindow.setWindowTitle('Изменение учетной записи пользователя')
            ubox.addRow('&Пароль:', passField)
            fbox.addRow(ubox)
            extraBox.addWidget(isAdmin)
            fbox.addRow(extraBox)

        else:
            self.accWindow.setWindowTitle('Удаление данных гостя')

        self.fName = QtWidgets.QLineEdit()
        self.fLastName = QtWidgets.QLineEdit()
        self.fSecondName = QtWidgets.QLineEdit()
        fbox.addRow('&Фамилия:', self.fLastName)
        fbox.addRow('&Имя:', self.fName)
        fbox.addRow('&Отчество:', self.fSecondName)
        hbtn = QtWidgets.QHBoxLayout()
        btnFnd = QtWidgets.QPushButton('&Найти')
        btnOK = QtWidgets.QPushButton('&OK')
        btnCancel = QtWidgets.QPushButton('&Cancel')
        hbtn.addWidget(btnFnd)
        hbtn.addWidget(btnOK)
        hbtn.addWidget(btnCancel)
        fbox.addRow(hbtn)
        btnCancel.clicked.connect(self.accWindow.close)
        self.btnBrowse.clicked.connect(self.browse_file)
        # btnOK.clicked.connect(сохраняем фото в папку /known , проверяем наличие логина, если нет то добавляем данные в БД)
        self.accWindow.setLayout(fbox)
        self.accWindow.adjustSize()
        self.accWindow.show()
        self.accWindow.exec_()

    def isValid(self, log, passw):
            print(log + " & " + passw)
            prio = 1
            if log == 'login1' and passw == 'password1':
                prio = 2
                self.unlock_menus(prio)
                self.dialog.close()
            elif log == 'login2' and passw == 'password2':
                self.unlock_menus(prio)
                self.dialog.close()
            else:
                pop = QtWidgets.QMessageBox()
                pop.setWindowTitle('Ошибка авторизации!')
                pop.setText('Неверные имя пользователя или пароль')
                pop.adjustSize()
                pop.show()
                pop.exec_()

    def browse_folder(self):
        self.listWidget             # temp item
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose directory")        #temp item

        if directory:                                           # temp item
            for file_name in os.listdir(directory):
                self.listWidget.addItem(file_name)

    def browse_file(self):
        self.fPath
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл", '/home')[0]
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
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1800, 700, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FRaVGApp()

    window.show()
    app.exec_()


if __name__ == '__main__':
    main()

