from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QDoubleValidator, QValidator
import pyqtgraph as pg
import tap_lib.Cooling as cooling


class TemperatureLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super(TemperatureLineEdit, self).__init__(parent)

        self.widg = parent
        self.editingFinished.connect(self.validating)

    def validating(self):
        validation_rule = QDoubleValidator(0, 2 ** 31 - 1, 10)

        if validation_rule.validate(self.text(), 0)[0] == QValidator.State.Acceptable:
            self.setFocus()
            self.widg.win.validate_values(self)
        else:
            self.setText('')


class IterationLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super(IterationLineEdit, self).__init__(parent)

        self.widg = parent
        self.editingFinished.connect(self.validating)

    def validating(self):
        validation_rule = QDoubleValidator(1, 2 ** 31 - 1, 0)

        if validation_rule.validate(self.text(), 0)[0] == QValidator.State.Acceptable:
            self.setFocus()
            self.widg.win.validate_values(self)
        else:
            self.setText('')


class CoolTypeComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super(CoolTypeComboBox, self).__init__(parent)

        self.widg = parent
        self.activated.connect(self.validating)

    def validating(self):
        self.widg.win.validate_values(self)


class MyCentralWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, win=None):
        super(MyCentralWidget, self).__init__(parent)

        self.win = win


class UiMainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        self.centralwidget = MyCentralWidget(parent=MainWindow, win=self)
        self.centralwidget.setObjectName("centralwidget")

        """Editlines and combobox"""

        self.init_temp_line = TemperatureLineEdit(parent=self.centralwidget)
        self.init_temp_line.setGeometry(QtCore.QRect(600, 70, 113, 22))
        self.init_temp_line.setObjectName("init_temp_line")

        self.final_temp_line = TemperatureLineEdit(parent=self.centralwidget)
        self.final_temp_line.setGeometry(QtCore.QRect(600, 120, 113, 22))
        self.final_temp_line.setObjectName("final_temp_line")

        self.max_iter_line = IterationLineEdit(parent=self.centralwidget)
        self.max_iter_line.setGeometry(QtCore.QRect(600, 170, 113, 22))
        self.max_iter_line.setObjectName("max_iter_line")

        self.iters_line = IterationLineEdit(parent=self.centralwidget)
        self.iters_line.setGeometry(QtCore.QRect(600, 220, 113, 22))
        self.iters_line.setObjectName("iters_line")

        self.poly_line = IterationLineEdit(parent=self.centralwidget)
        self.poly_line.setGeometry(QtCore.QRect(600, 330, 113, 22))
        self.poly_line.setObjectName("poly_line")
        self.poly_line.hide()

        """ComboBox"""

        self.cool_types_box = CoolTypeComboBox(parent=self.centralwidget)
        self.cool_types_box.setGeometry(QtCore.QRect(600, 270, 111, 31))

        font = QtGui.QFont()
        font.setPointSize(8)
        self.cool_types_box.setFont(font)
        self.cool_types_box.setEditable(False)
        self.cool_types_box.setObjectName("cool_types_box")
        self.cool_types_box.addItem("")
        self.cool_types_box.addItem("")
        self.cool_types_box.addItem("")
        self.cool_types_box.addItem("")

        """Labels"""

        self.init_temp_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.init_temp_label.setGeometry(QtCore.QRect(600, 50, 111, 16))
        self.init_temp_label.setObjectName("init_temp_label")

        self.final_temp_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.final_temp_label.setGeometry(QtCore.QRect(600, 100, 111, 16))
        self.final_temp_label.setObjectName("final_temp_label")

        self.max_iter_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.max_iter_label.setGeometry(QtCore.QRect(600, 150, 121, 16))
        self.max_iter_label.setObjectName("max_iter_label")

        self.type_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.type_label.setGeometry(QtCore.QRect(610, 250, 101, 16))
        self.type_label.setObjectName("type_label")

        self.iters_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.iters_label.setGeometry(QtCore.QRect(570, 200, 191, 20))
        self.iters_label.setObjectName("iters_label")

        self.poly_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.poly_label.setGeometry(QtCore.QRect(590, 310, 131, 16))
        self.poly_label.setObjectName("poly_label")
        self.poly_label.hide()

        """Buttons"""

        self.b_cooling = QtWidgets.QPushButton(self.centralwidget)
        self.b_cooling.setObjectName(u"plot_cooling")
        self.b_cooling.setGeometry(QtCore.QRect(600, 470, 75, 24))

        self.browse_button = QtWidgets.QPushButton(self.centralwidget)
        self.browse_button.setObjectName(u"browse_file_button")
        self.browse_button.setGeometry(QtCore.QRect(600, 520, 75, 24))

        """Plot"""

        self.plot_cooling = pg.PlotWidget(parent=self.centralwidget)
        self.plot_cooling.setGeometry(QtCore.QRect(0, 0, 500, 500))
        self.plot_cooling.setBackground('w')
        self.__pen = pg.mkPen(color='b', width=5)
        self.plot_cooling.setTitle("Temperature in iteration")
        self.plot_cooling.setLabel("left", "Temperature")
        self.plot_cooling.setLabel("bottom", "Iterations")
        self.plot_cooling.showGrid(x=True, y=True)
        self.plot_cooling.hide()

        """self fields"""

        self.__data_path = ""


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.init_temp_label.setText(_translate("MainWindow", "Initial temperature"))
        self.final_temp_label.setText(_translate("MainWindow", "Final temperature"))
        self.max_iter_label.setText(_translate("MainWindow", "Number of iterations"))
        self.iters_label.setText(_translate("MainWindow", "Iterations in single temperature"))

        self.type_label.setText(_translate("MainWindow", "Coolong function"))
        self.cool_types_box.setItemText(0, _translate("MainWindow", "Linear"))
        self.cool_types_box.setItemText(1, _translate("MainWindow", "Polynomial"))
        self.cool_types_box.setItemText(2, _translate("MainWindow", "Exponential"))
        self.cool_types_box.setItemText(3, _translate("MainWindow", "Logarithmic"))

        self.poly_label.setText(_translate("MainWindow", "Degree of polynomial"))

        self.b_cooling.setText(QtCore.QCoreApplication.translate("MainWindow", u"Plot Cooling", None))
        self.b_cooling.clicked.connect(self.__plot_clicked)

        self.browse_button.setText(QtCore.QCoreApplication.translate("MainWindow", u"Browse file", None))
        self.browse_button.clicked.connect(self.__open_dialog_box)


    def validate_values(self, param):
        """
        Function validating values if they are logically correct
        @param param: object which finished editing
        @type param: LineEdit subclass object
        """
        if param == self.init_temp_line:
            if self.final_temp_line.text() != '':
                if float(self.init_temp_line.text()) < float(self.final_temp_line.text()):
                    self.init_temp_line.setText('')
        if param == self.final_temp_line:
            if self.init_temp_line.text() != '':
                if float(self.init_temp_line.text()) < float(self.final_temp_line.text()):
                    self.final_temp_line.setText('')
        if param == self.max_iter_line:
            if self.iters_line.text() != '':
                if int(self.max_iter_line.text()) < int(self.iters_line.text()):
                    self.max_iter_line.setText('')
        if param == self.iters_line:
            if self.max_iter_line.text() != '':
                if int(self.max_iter_line.text()) < int(self.iters_line.text()):
                    self.iters_line.setText('')
        if param == self.cool_types_box:
            if self.cool_types_box.currentText() == "Polynomial":
                self.poly_label.show()
                self.poly_line.show()
            else:
                self.poly_label.hide()
                self.poly_line.hide()

    def __plot_clicked(self):
        """
        Function checking if every necessary LineEdit is filled; if correct plots temperature graph
        """
        if all(txt != '' for txt in [self.init_temp_line.text(), self.final_temp_line.text(), self.max_iter_line.text(),
                                     self.iters_line.text()]):
            if self.cool_types_box.currentText() == "Polynominal" and self.poly_line == '':
                pass
            else:
                cool_type = list(cooling.CoolingTypes)[self.cool_types_box.currentIndex()]
                poly = self.poly_line.text()
                if poly == '':
                    poly = None
                else:
                    poly = int(self.poly_line.text())
                cool_f = cooling.cooling_factory(float(self.init_temp_line.text()), float(self.final_temp_line.text()),
                                            int(self.max_iter_line.text()), cool_type, int(self.iters_line.text()),
                                            poly)
                y = []
                temp = float(self.init_temp_line.text())
                for i in range(int(self.max_iter_line.text())):
                    temp = cool_f(temp, i)
                    y.append(temp)
                self.plot_cooling.plotItem.clear()
                self.plot_cooling.plot([i for i in range(int(self.max_iter_line.text()))], y, pen=self.__pen)
                self.plot_cooling.show()

    def __open_dialog_box(self):
        filename = QtWidgets.QFileDialog.getOpenFileName()
        msg = QtWidgets.QMessageBox()
        if filename[0].endswith('.csv'):
            self.__data_path = filename[0]
            msg.setText("Data loaded")
        else:
            msg.setWindowTitle("Error")
            msg.setText("Wrong file extension\nExpected .csv")
            msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)

        msg.exec()


if __name__ == "__main__":
    import sys

    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = UiMainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())