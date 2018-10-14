# ---------------------------------------------------------------#
# __name__ = "ImageRestoration_EE610_Assignment"
# __author__ = "Shyama P"
# __version__ = "1.0"
# __email__ = "183079031@iitb.ac.in"
# __status__ = "Development"
# ---------------------------------------------------------------#

# This code was generated using QT4 Desginer
# Only few modifications were done on this code

# PyQt4 libraries are used for GUI

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8

    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


class ImageRestorationGuiClass(object):
    def setupUi(self, MainWindow):
        decimal_validator = QtGui.QDoubleValidator()
        decimal_validator.setBottom(0.0)
        decimal_validator.setDecimals(3)
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(761, 553)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setStyleSheet(_fromUtf8("font: 11pt \"Dyuthi\";"))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_2 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.buttonOpen = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonOpen.sizePolicy().hasHeightForWidth())
        self.buttonOpen.setSizePolicy(sizePolicy)
        self.buttonOpen.setObjectName(_fromUtf8("buttonOpen"))
        self.horizontalLayout_5.addWidget(self.buttonOpen)
        self.buttonSave = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonSave.sizePolicy().hasHeightForWidth())
        self.buttonSave.setSizePolicy(sizePolicy)
        self.buttonSave.setObjectName(_fromUtf8("buttonSave"))
        self.horizontalLayout_5.addWidget(self.buttonSave)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.line_2 = QtGui.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtGui.QFrame.HLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.verticalLayout.addWidget(self.line_2)
        self.labelBlurName = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelBlurName.sizePolicy().hasHeightForWidth())
        self.labelBlurName.setSizePolicy(sizePolicy)
        self.labelBlurName.setObjectName(_fromUtf8("labelBlurName"))
        self.verticalLayout.addWidget(self.labelBlurName)
        self.comboBoxKernel = QtGui.QComboBox(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBoxKernel.sizePolicy().hasHeightForWidth())
        self.comboBoxKernel.setSizePolicy(sizePolicy)
        self.comboBoxKernel.setObjectName(_fromUtf8("comboBoxKernel"))
        self.comboBoxKernel.addItem(_fromUtf8(""))
        self.comboBoxKernel.addItem(_fromUtf8(""))
        self.comboBoxKernel.addItem(_fromUtf8(""))
        self.comboBoxKernel.addItem(_fromUtf8(""))
        self.comboBoxKernel.addItem(_fromUtf8(""))
        self.verticalLayout.addWidget(self.comboBoxKernel)
        self.labelKernelDisplay = QtGui.QLabel(self.centralwidget)
        self.labelKernelDisplay.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelKernelDisplay.sizePolicy().hasHeightForWidth())
        self.labelKernelDisplay.setSizePolicy(sizePolicy)
        self.labelKernelDisplay.setMinimumSize(QtCore.QSize(21, 21))
        self.labelKernelDisplay.setMaximumSize(QtCore.QSize(200, 200))
        self.labelKernelDisplay.setText(_fromUtf8(""))
        self.labelKernelDisplay.setAlignment(QtCore.Qt.AlignCenter)
        self.labelKernelDisplay.setObjectName(_fromUtf8("labelKernelDisplay"))
        self.verticalLayout.addWidget(self.labelKernelDisplay)
        self.line = QtGui.QFrame(self.centralwidget)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.verticalLayout.addWidget(self.line)
        self.buttonFullInv = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonFullInv.sizePolicy().hasHeightForWidth())
        self.buttonFullInv.setSizePolicy(sizePolicy)
        self.buttonFullInv.setObjectName(_fromUtf8("buttonFullInv"))
        self.verticalLayout.addWidget(self.buttonFullInv)
        self.line_3 = QtGui.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtGui.QFrame.HLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.verticalLayout.addWidget(self.line_3)
        self.buttonInv = QtGui.QPushButton(self.centralwidget)
        self.buttonInv.setObjectName(_fromUtf8("buttonInv"))
        self.verticalLayout.addWidget(self.buttonInv)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_11 = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setTextFormat(QtCore.Qt.PlainText)
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.horizontalLayout.addWidget(self.label_11)
        self.input_radius = QtGui.QLineEdit(self.centralwidget)
        int_validator = QtGui.QIntValidator()
        int_validator.setBottom(1)
        self.input_radius.setValidator(int_validator)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.input_radius.sizePolicy().hasHeightForWidth())
        self.input_radius.setSizePolicy(sizePolicy)
        self.input_radius.setObjectName(_fromUtf8("input_radius"))
        self.horizontalLayout.addWidget(self.input_radius)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.line_4 = QtGui.QFrame(self.centralwidget)
        self.line_4.setFrameShape(QtGui.QFrame.HLine)
        self.line_4.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_4.setObjectName(_fromUtf8("line_4"))
        self.verticalLayout.addWidget(self.line_4)
        self.buttonWeiner = QtGui.QPushButton(self.centralwidget)
        self.buttonWeiner.setObjectName(_fromUtf8("buttonWeiner"))
        self.verticalLayout.addWidget(self.buttonWeiner)
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setMinimumSize(QtCore.QSize(0, 0))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout_8.addWidget(self.label_4)
        self.input_K = QtGui.QLineEdit(self.centralwidget)
        self.input_K.setValidator(decimal_validator)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.input_K.sizePolicy().hasHeightForWidth())
        self.input_K.setSizePolicy(sizePolicy)
        self.input_K.setObjectName(_fromUtf8("input_K"))
        self.horizontalLayout_8.addWidget(self.input_K)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.line_5 = QtGui.QFrame(self.centralwidget)
        self.line_5.setFrameShape(QtGui.QFrame.HLine)
        self.line_5.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_5.setObjectName(_fromUtf8("line_5"))
        self.verticalLayout.addWidget(self.line_5)
        self.buttonCLS = QtGui.QPushButton(self.centralwidget)
        self.buttonCLS.setObjectName(_fromUtf8("buttonCLS"))
        self.verticalLayout.addWidget(self.buttonCLS)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.label100 = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label100.sizePolicy().hasHeightForWidth())
        self.label100.setSizePolicy(sizePolicy)
        self.label100.setObjectName(_fromUtf8("label100"))
        self.horizontalLayout_9.addWidget(self.label100)
        self.input_gamma = QtGui.QLineEdit(self.centralwidget)
        self.input_gamma.setValidator(decimal_validator)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.input_gamma.sizePolicy().hasHeightForWidth())
        self.input_gamma.setSizePolicy(sizePolicy)
        self.input_gamma.setObjectName(_fromUtf8("input_gamma"))
        self.horizontalLayout_9.addWidget(self.input_gamma)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.line_6 = QtGui.QFrame(self.centralwidget)
        self.line_6.setFrameShape(QtGui.QFrame.HLine)
        self.line_6.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_6.setObjectName(_fromUtf8("line_6"))
        self.verticalLayout.addWidget(self.line_6)
        self.buttonPSNR = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonPSNR.sizePolicy().hasHeightForWidth())
        self.buttonPSNR.setSizePolicy(sizePolicy)
        self.buttonPSNR.setObjectName(_fromUtf8("buttonPSNR"))
        self.verticalLayout.addWidget(self.buttonPSNR)
        self.buttonSSIM = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonSSIM.sizePolicy().hasHeightForWidth())
        self.buttonSSIM.setSizePolicy(sizePolicy)
        self.buttonSSIM.setObjectName(_fromUtf8("buttonSSIM"))
        self.verticalLayout.addWidget(self.buttonSSIM)

        self.line_24 = QtGui.QFrame(self.centralwidget)
        self.line_24.setFrameShape(QtGui.QFrame.HLine)
        self.line_24.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_24.setObjectName(_fromUtf8("line_24"))
        self.verticalLayout.addWidget(self.line_24)
        self.buttonTrueImage = QtGui.QPushButton(self.centralwidget)
        self.buttonTrueImage.setObjectName(_fromUtf8("buttonTrueImage"))
        self.verticalLayout.addWidget(self.buttonTrueImage)
        self.buttonClearTrueImage = QtGui.QPushButton(self.centralwidget)
        self.buttonClearTrueImage.setObjectName(_fromUtf8("buttonClearTrueImage"))
        self.verticalLayout.addWidget(self.buttonClearTrueImage)

        self.line_11 = QtGui.QFrame(self.centralwidget)
        self.line_11.setFrameShape(QtGui.QFrame.HLine)
        self.line_11.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_11.setObjectName(_fromUtf8("line_11"))
        self.verticalLayout.addWidget(self.line_11)
        # spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        # self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.line_7 = QtGui.QFrame(self.centralwidget)
        self.line_7.setFrameShape(QtGui.QFrame.VLine)
        self.line_7.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_7.setObjectName(_fromUtf8("line_7"))
        self.horizontalLayout_3.addWidget(self.line_7)
        self.verticalLayout_5 = QtGui.QVBoxLayout()
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.horizontalLayout_11 = QtGui.QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        self.labelIn = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelIn.sizePolicy().hasHeightForWidth())
        self.labelIn.setSizePolicy(sizePolicy)
        self.labelIn.setFrameShape(QtGui.QFrame.NoFrame)
        self.labelIn.setText(_fromUtf8(""))
        self.labelIn.setObjectName(_fromUtf8("labelIn"))
        self.horizontalLayout_11.addWidget(self.labelIn)
        self.line_9 = QtGui.QFrame(self.centralwidget)
        self.line_9.setFrameShape(QtGui.QFrame.VLine)
        self.line_9.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_9.setObjectName(_fromUtf8("line_9"))
        self.horizontalLayout_11.addWidget(self.line_9)
        self.labelOut = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelOut.sizePolicy().hasHeightForWidth())
        self.labelOut.setSizePolicy(sizePolicy)
        self.labelOut.setFrameShape(QtGui.QFrame.NoFrame)
        self.labelOut.setFrameShadow(QtGui.QFrame.Sunken)
        self.labelOut.setText(_fromUtf8(""))
        self.labelOut.setObjectName(_fromUtf8("labelOut"))
        self.horizontalLayout_11.addWidget(self.labelOut)
        self.verticalLayout_5.addLayout(self.horizontalLayout_11)
        self.line_13 = QtGui.QFrame(self.centralwidget)
        self.line_13.setFrameShape(QtGui.QFrame.HLine)
        self.line_13.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_13.setObjectName(_fromUtf8("line_13"))
        self.verticalLayout_5.addWidget(self.line_13)
        self.horizontalLayout_12 = QtGui.QHBoxLayout()
        self.horizontalLayout_12.setObjectName(_fromUtf8("horizontalLayout_12"))
        self.label_7 = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Dyuthi"))
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.horizontalLayout_12.addWidget(self.label_7)
        self.line_10 = QtGui.QFrame(self.centralwidget)
        self.line_10.setFrameShape(QtGui.QFrame.VLine)
        self.line_10.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_10.setObjectName(_fromUtf8("line_10"))
        self.horizontalLayout_12.addWidget(self.line_10)
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.horizontalLayout_12.addWidget(self.label_6)
        self.verticalLayout_5.addLayout(self.horizontalLayout_12)
        self.line_8 = QtGui.QFrame(self.centralwidget)
        self.line_8.setFrameShape(QtGui.QFrame.HLine)
        self.line_8.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_8.setObjectName(_fromUtf8("line_8"))
        self.verticalLayout_5.addWidget(self.line_8)
        self.horizontalLayout_10 = QtGui.QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.label_88 = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_88.sizePolicy().hasHeightForWidth())
        self.label_88.setSizePolicy(sizePolicy)
        self.label_88.setObjectName(_fromUtf8("label_88"))
        self.horizontalLayout_6.addWidget(self.label_88)
        self.label_og_psnr = QtGui.QLabel(self.centralwidget)
        self.label_og_psnr.setObjectName(_fromUtf8("label_og_psnr"))
        self.horizontalLayout_6.addWidget(self.label_og_psnr)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.horizontalLayout_4.addWidget(self.label_5)
        self.label_og_ssim = QtGui.QLabel(self.centralwidget)
        self.label_og_ssim.setObjectName(_fromUtf8("label_og_ssim"))
        self.horizontalLayout_4.addWidget(self.label_og_ssim)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_10.addLayout(self.verticalLayout_3)
        self.line_14 = QtGui.QFrame(self.centralwidget)
        self.line_14.setFrameShape(QtGui.QFrame.VLine)
        self.line_14.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_14.setObjectName(_fromUtf8("line_14"))
        self.horizontalLayout_10.addWidget(self.line_14)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_2.addWidget(self.label_2)
        self.label_res_psnr = QtGui.QLabel(self.centralwidget)
        self.label_res_psnr.setObjectName(_fromUtf8("label_res_psnr"))
        self.horizontalLayout_2.addWidget(self.label_res_psnr)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.label_12 = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.horizontalLayout_7.addWidget(self.label_12)
        self.label_res_ssim = QtGui.QLabel(self.centralwidget)
        self.label_res_ssim.setObjectName(_fromUtf8("label_res_ssim"))
        self.horizontalLayout_7.addWidget(self.label_res_ssim)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_10.addLayout(self.verticalLayout_2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_3.addLayout(self.verticalLayout_5)
        self.gridLayout_2.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtGui.QAction(MainWindow)
        self.actionOpen.setObjectName(_fromUtf8("actionOpen"))
        self.actionSave = QtGui.QAction(MainWindow)
        self.actionSave.setObjectName(_fromUtf8("actionSave"))
        self.actionQuit = QtGui.QAction(MainWindow)
        self.actionQuit.setObjectName(_fromUtf8("actionQuit"))
        self.actionUndo_All_Changes = QtGui.QAction(MainWindow)
        self.actionUndo_All_Changes.setObjectName(_fromUtf8("actionUndo_All_Changes"))
        self.actionAbout = QtGui.QAction(MainWindow)
        self.actionAbout.setObjectName(_fromUtf8("actionAbout"))

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "EE610_ImageRestoration", None))
        self.buttonOpen.setText(_translate("MainWindow", "Open", None))
        self.buttonSave.setText(_translate("MainWindow", "Save", None))
        self.labelBlurName.setText(_translate("MainWindow", "Choose Blur Kernel", None))
        self.comboBoxKernel.setItemText(0, _translate("MainWindow", "Blur Kernel 1", None))
        self.comboBoxKernel.setItemText(1, _translate("MainWindow", "Blur Kernel 2", None))
        self.comboBoxKernel.setItemText(2, _translate("MainWindow", "Blur Kernel 3", None))
        self.comboBoxKernel.setItemText(3, _translate("MainWindow", "Blur Kernel 4", None))
        self.comboBoxKernel.setItemText(4, _translate("MainWindow", "Estimated Kernel", None))
        self.buttonFullInv.setText(_translate("MainWindow", "Full Inverse Filter", None))
        self.buttonInv.setText(_translate("MainWindow", "Truncated Inverse Filter", None))
        self.label_11.setText(_translate("MainWindow", "Select radius : ", None))
        self.input_radius.setPlaceholderText(_translate("MainWindow", "Enter radius", None))
        self.buttonWeiner.setText(_translate("MainWindow", "Weiner Filter", None))
        self.label_4.setText(_translate("MainWindow", "Select K          : ", None))
        self.input_K.setPlaceholderText(_translate("MainWindow", "Enter K value", None))
        self.buttonCLS.setText(_translate("MainWindow", "Constrained LS Filter", None))
        self.label100.setText(_translate("MainWindow", "Select gamma: ", None))
        self.input_gamma.setPlaceholderText(_translate("MainWindow", "Enter gamma", None))
        self.buttonPSNR.setText(_translate("MainWindow", "Compute PSNR", None))
        self.buttonSSIM.setText(_translate("MainWindow", "Compute SSIM", None))
        self.buttonTrueImage.setText(_translate("MainWindow", "Load true image", None))
        self.buttonClearTrueImage.setText(_translate("MainWindow", "Clear true image", None))
        self.label_7.setText(_translate("MainWindow", "Original Image", None))
        self.label_6.setText(_translate("MainWindow", "Restored Image", None))
        self.label_88.setText(_translate("MainWindow", "PSNR : ", None))
        self.label_og_psnr.setText(_translate("MainWindow", "--", None))
        self.label_5.setText(_translate("MainWindow", "SSIM : ", None))
        self.label_og_ssim.setText(_translate("MainWindow", "--", None))
        self.label_2.setText(_translate("MainWindow", "PSNR : ", None))
        self.label_res_psnr.setText(_translate("MainWindow", "--", None))
        self.label_12.setText(_translate("MainWindow", "SSIM : ", None))
        self.label_res_ssim.setText(_translate("MainWindow", "--", None))
        self.actionOpen.setText(_translate("MainWindow", "Open", None))
        self.actionSave.setText(_translate("MainWindow", "Save", None))
        self.actionQuit.setText(_translate("MainWindow", "Quit", None))
        self.actionUndo_All_Changes.setText(_translate("MainWindow", "Undo All Changes", None))
        self.actionAbout.setText(_translate("MainWindow", "About", None))

