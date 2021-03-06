# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(480, 320))
        MainWindow.setFocusPolicy(QtCore.Qt.NoFocus)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        MainWindow.setAcceptDrops(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 800, 600))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(5, -1, 5, -1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.imgLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.imgLabel.setGeometry(5, 1, 800, 600)
        self.imgLabel.setMinimumSize(QtCore.QSize(4, 0))
        self.imgLabel.setMaximumSize(QtCore.QSize(1900, 1080))
        self.imgLabel.resize(800, 600)
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        self.horizontalLayout_2.addWidget(self.imgLabel)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 30))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuVideo = QtWidgets.QMenu(self.menubar)
        self.menuVideo.setObjectName("menuVideo")
        self.menuZoom = QtWidgets.QMenu(self.menuVideo)
        self.menuZoom.setObjectName("menuZoom")
        self.menuAspect_Ratio = QtWidgets.QMenu(self.menuVideo)
        self.menuAspect_Ratio.setObjectName("menuAspect_Ratio")
        self.menuCrop = QtWidgets.QMenu(self.menuVideo)
        self.menuCrop.setObjectName("menuCrop")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuTools")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setStatusTip("")
        self.actionOpen.setObjectName("actionOpen")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionTake_Snapshot = QtWidgets.QAction(MainWindow)
        self.actionTake_Snapshot.setObjectName("actionTake_Snapshot")
        self.actionPreferences = QtWidgets.QAction(MainWindow)
        self.actionPreferences.setObjectName("actionPreferences")
        self.actionMinimal_Interface = QtWidgets.QAction(MainWindow)
        self.actionMinimal_Interface.setCheckable(True)
        self.actionMinimal_Interface.setObjectName("actionMinimal_Interface")
        self.actionFullscreen_Interface = QtWidgets.QAction(MainWindow)
        self.actionFullscreen_Interface.setCheckable(True)
        self.actionFullscreen_Interface.setObjectName("actionFullscreen_Interface")
        self.actionAdvanced_Controls = QtWidgets.QAction(MainWindow)
        self.actionAdvanced_Controls.setCheckable(True)
        self.actionAdvanced_Controls.setObjectName("actionAdvanced_Controls")
        self.actionStatus_Bar = QtWidgets.QAction(MainWindow)
        self.actionStatus_Bar.setCheckable(True)
        self.actionStatus_Bar.setObjectName("actionStatus_Bar")
        self.actionHelp = QtWidgets.QAction(MainWindow)
        self.actionHelp.setObjectName("actionHelp")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.GroupZoom = QtWidgets.QActionGroup(MainWindow)
        self.GroupZoom.setObjectName("GroupZoom")
        self.action1_4_Quarter = QtWidgets.QAction(self.GroupZoom)
        self.action1_4_Quarter.setCheckable(True)
        self.action1_4_Quarter.setChecked(False)
        self.action1_4_Quarter.setObjectName("action1_4_Quarter")
        self.action1_2_Half = QtWidgets.QAction(self.GroupZoom)
        self.action1_2_Half.setCheckable(True)
        self.action1_2_Half.setChecked(False)
        self.action1_2_Half.setObjectName("action1_2_Half")
        self.action1_1_Original = QtWidgets.QAction(self.GroupZoom)
        self.action1_1_Original.setCheckable(True)
        self.action1_1_Original.setChecked(True)
        self.action1_1_Original.setObjectName("action1_1_Original")
        self.action2_1_Double = QtWidgets.QAction(self.GroupZoom)
        self.action2_1_Double.setCheckable(True)
        self.action2_1_Double.setChecked(False)
        self.action2_1_Double.setObjectName("action2_1_Double")
        self.GroupAspectRatio = QtWidgets.QActionGroup(MainWindow)
        self.GroupAspectRatio.setObjectName("GroupAspectRatio")
        self.actionDefault = QtWidgets.QAction(self.GroupAspectRatio)
        self.actionDefault.setCheckable(True)
        self.actionDefault.setChecked(True)
        self.actionDefault.setObjectName("actionDefault")
        self.action16_9 = QtWidgets.QAction(self.GroupAspectRatio)
        self.action16_9.setCheckable(True)
        self.action16_9.setObjectName("action16_9")
        self.action4_3 = QtWidgets.QAction(self.GroupAspectRatio)
        self.action4_3.setCheckable(True)
        self.action4_3.setObjectName("action4_3")
        self.action1_1 = QtWidgets.QAction(self.GroupAspectRatio)
        self.action1_1.setCheckable(True)
        self.action1_1.setObjectName("action1_1")
        self.action5_4 = QtWidgets.QAction(self.GroupAspectRatio)
        self.action5_4.setCheckable(True)
        self.action5_4.setObjectName("action5_4")
        self.GroupCrop = QtWidgets.QActionGroup(MainWindow)
        self.GroupCrop.setObjectName("GroupCrop")
        self.actionDefault_2 = QtWidgets.QAction(self.GroupCrop)
        self.actionDefault_2.setCheckable(True)
        self.actionDefault_2.setChecked(True)
        self.actionDefault_2.setObjectName("actionDefault_2")
        self.action16_10 = QtWidgets.QAction(self.GroupCrop)
        self.action16_10.setCheckable(True)
        self.action16_10.setObjectName("action16_10")
        self.action4_4 = QtWidgets.QAction(self.GroupCrop)
        self.action4_4.setCheckable(True)
        self.action4_4.setObjectName("action4_4")
        self.action1_2 = QtWidgets.QAction(self.GroupCrop)
        self.action1_2.setCheckable(True)
        self.action1_2.setObjectName("action1_2")
        self.action5_5 = QtWidgets.QAction(self.GroupCrop)
        self.action5_5.setCheckable(True)
        self.action5_5.setObjectName("action5_5")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionQuit)
        self.menuZoom.addAction(self.action1_4_Quarter)
        self.menuZoom.addAction(self.action1_2_Half)
        self.menuZoom.addAction(self.action1_1_Original)
        self.menuZoom.addAction(self.action2_1_Double)
        self.menuAspect_Ratio.addAction(self.actionDefault)
        self.menuAspect_Ratio.addAction(self.action16_9)
        self.menuAspect_Ratio.addAction(self.action4_3)
        self.menuAspect_Ratio.addAction(self.action1_1)
        self.menuAspect_Ratio.addAction(self.action5_4)
        self.menuCrop.addAction(self.actionDefault_2)
        self.menuCrop.addAction(self.action16_10)
        self.menuCrop.addAction(self.action4_4)
        self.menuCrop.addAction(self.action1_2)
        self.menuCrop.addAction(self.action5_5)
        self.menuVideo.addAction(self.menuZoom.menuAction())
        self.menuVideo.addAction(self.menuAspect_Ratio.menuAction())
        self.menuVideo.addAction(self.menuCrop.menuAction())
        self.menuVideo.addSeparator()
        self.menuVideo.addAction(self.actionTake_Snapshot)
        self.menuTools.addAction(self.actionPreferences)
        self.menuView.addAction(self.actionMinimal_Interface)
        self.menuView.addAction(self.actionFullscreen_Interface)
        self.menuView.addAction(self.actionAdvanced_Controls)
        self.menuView.addAction(self.actionStatus_Bar)
        self.menuHelp.addAction(self.actionHelp)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuVideo.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ML Facemask"))
        
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuVideo.setTitle(_translate("MainWindow", "Video"))
        self.menuZoom.setTitle(_translate("MainWindow", "Zoom"))
        self.menuAspect_Ratio.setTitle(_translate("MainWindow", "Aspect Ratio"))
        self.menuCrop.setTitle(_translate("MainWindow", "Crop"))
        self.menuTools.setTitle(_translate("MainWindow", "Tools"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionQuit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionTake_Snapshot.setText(_translate("MainWindow", "Take Snapshot"))
        self.actionTake_Snapshot.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionPreferences.setText(_translate("MainWindow", "Preferences"))
        self.actionMinimal_Interface.setText(_translate("MainWindow", "Minimal Interface"))
        self.actionMinimal_Interface.setShortcut(_translate("MainWindow", "Ctrl+H"))
        self.actionFullscreen_Interface.setText(_translate("MainWindow", "Fullscreen Interface"))
        self.actionFullscreen_Interface.setShortcut(_translate("MainWindow", "F11"))
        self.actionAdvanced_Controls.setText(_translate("MainWindow", "Advanced Controls"))
        self.actionStatus_Bar.setText(_translate("MainWindow", "Status Bar"))
        self.actionHelp.setText(_translate("MainWindow", "Help"))
        self.actionHelp.setShortcut(_translate("MainWindow", "F1"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionAbout.setShortcut(_translate("MainWindow", "Shift+F1"))
        self.action1_4_Quarter.setText(_translate("MainWindow", "1:4 Quarter"))
        self.action1_2_Half.setText(_translate("MainWindow", "1:2 Half"))
        self.action1_1_Original.setText(_translate("MainWindow", "1:1 Original"))
        self.action2_1_Double.setText(_translate("MainWindow", "2:1 Double"))
        self.actionDefault.setText(_translate("MainWindow", "Default"))
        self.action16_9.setText(_translate("MainWindow", "16:9"))
        self.action4_3.setText(_translate("MainWindow", "4:3"))
        self.action1_1.setText(_translate("MainWindow", "1:1"))
        self.action5_4.setText(_translate("MainWindow", "5:4"))
        self.actionDefault_2.setText(_translate("MainWindow", "Default"))
        self.action16_10.setText(_translate("MainWindow", "16:9"))
        self.action4_4.setText(_translate("MainWindow", "4:3"))
        self.action1_2.setText(_translate("MainWindow", "1:1"))
        self.action5_5.setText(_translate("MainWindow", "5:4"))
