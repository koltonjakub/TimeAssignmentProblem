from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QDoubleValidator, QValidator
import pyqtgraph as pg
import Cooling
from Solver import Solver
import Factory as fac
from typing import Callable
import Probability as prob
import os


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


class UiResultWindow(object):
    def __init__(self, path: str, cool_fcn: Callable[[float, int], float], init_temp: float, max_iter: int):
        self.rel_improvement_label = None
        self.abs_improvement_label = None
        self.best_obj_label = None
        self.plot_val = None
        self.plot_best_val = None
        self.plot_probability = None
        self.__pen = None
        self.__prob_pen = None
        self.plot_cooling = None
        self.statusbar = None
        self.menubar = None
        self.centralwidget = None
        self.path = path
        self.cool = cool_fcn
        self.init_temp = init_temp
        self.max_iter = max_iter

        dir = os.getcwd()
        par = os.path.dirname(dir)

        solver = Solver(cost=fac.get_cost, sol_gen=fac.random_neighbour, cool=self.cool, probability=prob.exponential,
                        init_temp=self.init_temp, max_iterations=self.max_iter,
                        log_file_path=os.path.join(par, "logs", "log_junk", "junk.log"),
                        csv_file_path=os.path.join(par, "tst_algorithm_properties", "results", "test_gui.csv"))

        solver.SolutionType = fac.FactoryAssignmentSchedule
        solver.probability = prob.exponential

        solver.init_sol = fac.generate_starting_solution(self.path)
        solver.log_results = True
        solver.remember_visited_solution = False

        self.solution, self.scope = solver.simulate_annealing()

        init_val = self.scope.best_cost_function[0]
        best_val = self.scope.best_cost_function[-1]

        self.best_obj_val = round(best_val, 2)
        self.abs_improvement = round(init_val - best_val, 2)
        self.rel_improvement = round(100*(init_val - best_val)/init_val if best_val != 0 else 1, 2)


    def setupUi(self, ResultWindow):
        if not ResultWindow.objectName():
            ResultWindow.setObjectName(u"ResultWindow")
        ResultWindow.resize(1000, 700)
        self.centralwidget = QtWidgets.QWidget(ResultWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        ResultWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ResultWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        ResultWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        ResultWindow.setStatusBar(self.statusbar)

        """Labels"""

        self.best_obj_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.best_obj_label.setGeometry(QtCore.QRect(100, 620, 300, 16))
        self.best_obj_label.setObjectName("best_obj_label")

        self.abs_improvement_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.abs_improvement_label.setGeometry(QtCore.QRect(100, 640, 300, 16))
        self.abs_improvement_label.setObjectName("abs_improv")

        self.rel_improvement_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.rel_improvement_label.setGeometry(QtCore.QRect(100, 660, 300, 16))
        self.rel_improvement_label.setObjectName("rel_improv")

        self.retranslateUi(ResultWindow)

        QtCore.QMetaObject.connectSlotsByName(ResultWindow)

        "Plots"

        self.__pen = pg.mkPen(color='b', width=5)

        self.plot_cooling = pg.PlotWidget(parent=self.centralwidget)
        self.plot_cooling.setGeometry(QtCore.QRect(0, 0, 500, 300))
        self.plot_cooling.setBackground('w')
        self.__pen = pg.mkPen(color='b', width=5)
        self.plot_cooling.setTitle('<span style="color: black; font-size: 18px">Temperature</span>')
        self.plot_cooling.setLabel("left", '<span style="color: black; font-size: 18px">Temperature</span>')
        self.plot_cooling.setLabel("bottom", '<span style="color: black; font-size: 18px">Iteration</span>')
        self.plot_cooling.showGrid(x=True, y=True)

        self.plot_cooling.plotItem.clear()
        self.plot_cooling.plot(self.scope.iteration, self.scope.temperature, pen=self.__pen)

        self.plot_cooling.getAxis('left').setPen('black')
        self.plot_cooling.getAxis('left').setTextPen('black')
        self.plot_cooling.getAxis('bottom').setPen('black')
        self.plot_cooling.getAxis('bottom').setTextPen('black')


        self.plot_probability = pg.PlotWidget(parent=self.centralwidget)
        self.plot_probability.setGeometry(QtCore.QRect(500, 0, 500, 300))
        self.plot_probability.setBackground('w')
        self.__prob_pen = pg.mkPen(color='w', width=0, symbol='.')
        self.plot_probability.setTitle('<span style="color: black; font-size: 18px">Probability</span>')
        self.plot_probability.setLabel("left", '<span style="color: black; font-size: 18px">Probability</span>')
        self.plot_probability.setLabel("bottom", '<span style="color: black; font-size: 18px">Iteration</span>')
        self.plot_probability.showGrid(x=True, y=True)

        self.plot_probability.plotItem.clear()
        self.plot_probability.plot(self.scope.iteration, self.scope.probability_of_transition, pen=self.__prob_pen,
                                   symbolSize=5, symbolBrush='b')

        self.plot_probability.getAxis('left').setPen('black')
        self.plot_probability.getAxis('left').setTextPen('black')
        self.plot_probability.getAxis('bottom').setPen('black')
        self.plot_probability.getAxis('bottom').setTextPen('black')


        self.plot_best_val = pg.PlotWidget(parent=self.centralwidget)
        self.plot_best_val.setGeometry(QtCore.QRect(0, 300, 500, 300))
        self.plot_best_val.setBackground('w')
        self.plot_best_val.setTitle('<span style="color: black; font-size: 18px">Best objective value</span>')
        self.plot_best_val.setLabel("left", '<span style="color: black; font-size: 18px">Best objective value</span>')
        self.plot_best_val.setLabel("bottom", '<span style="color: black; font-size: 18px">Iteration</span>')
        self.plot_best_val.showGrid(x=True, y=True)

        self.plot_best_val.plotItem.clear()
        self.plot_best_val.plot(self.scope.iteration, self.scope.best_cost_function, pen=self.__pen)

        self.plot_best_val.getAxis('left').setPen('black')
        self.plot_best_val.getAxis('left').setTextPen('black')
        self.plot_best_val.getAxis('bottom').setPen('black')
        self.plot_best_val.getAxis('bottom').setTextPen('black')

        self.plot_val = pg.PlotWidget(parent=self.centralwidget)
        self.plot_val.setGeometry(QtCore.QRect(500, 300, 500, 300))
        self.plot_val.setBackground('w')
        self.plot_val.setTitle('<span style="color: black; font-size: 18px">Objective value</span>')
        self.plot_val.setLabel("left", '<span style="color: black; font-size: 18px">Objective value</span>')
        self.plot_val.setLabel("bottom", '<span style="color: black; font-size: 18px">Iteration</span>')
        self.plot_val.showGrid(x=True, y=True)

        self.plot_val.plotItem.clear()
        self.plot_val.plot(self.scope.iteration, self.scope.cost_function, pen=self.__pen)

        self.plot_val.getAxis('left').setPen('black')
        self.plot_val.getAxis('left').setTextPen('black')
        self.plot_val.getAxis('bottom').setPen('black')
        self.plot_val.getAxis('bottom').setTextPen('black')



    def retranslateUi(self, ResultWindow):
        _translate = QtCore.QCoreApplication.translate
        ResultWindow.setWindowTitle(QtCore.QCoreApplication.translate("ResultWindow", u"ResultWindow", None))

        self.best_obj_label.setText(_translate(
            "ResultWindow", "Best objective value: "+str(self.best_obj_val)))
        self.abs_improvement_label.setText(_translate(
            "ResultWindow", "Absolute improvement: "+str(self.abs_improvement)))
        self.rel_improvement_label.setText(_translate(
            "ResultWindow", "Relative improvement: "+str(self.rel_improvement)+"%"))




class UiMainWindow(object):
    def __init__(self):
        self.statusbar = None
        self.menubar = None
        self.__data_path = None
        self.__pen = None
        self.plot_cooling = None
        self.run_button = None
        self.browse_button = None
        self.b_cooling = None
        self.poly_label = None
        self.iters_label = None
        self.type_label = None
        self.max_iter_label = None
        self.final_temp_label = None
        self.init_temp_label = None
        self.cool_types_box = None
        self.poly_line = None
        self.iters_line = None
        self.max_iter_line = None
        self.final_temp_line = None
        self.init_temp_line = None
        self.centralwidget = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        self.centralwidget = MyCentralWidget(parent=MainWindow, win=self)
        self.centralwidget.setObjectName("centralwidget")

        """Editlines"""

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
        self.b_cooling.setObjectName(u"b_cooling")
        self.b_cooling.setGeometry(QtCore.QRect(600, 400, 75, 24))

        self.browse_button = QtWidgets.QPushButton(self.centralwidget)
        self.browse_button.setObjectName(u"browse_file_button")
        self.browse_button.setGeometry(QtCore.QRect(600, 450, 75, 24))

        self.run_button = QtWidgets.QPushButton(self.centralwidget)
        self.run_button.setText(u"run_button")
        self.run_button.setGeometry(QtCore.QRect(600, 500, 90, 24))

        """Plot"""

        self.plot_cooling = pg.PlotWidget(parent=self.centralwidget)
        self.plot_cooling.setGeometry(QtCore.QRect(0, 0, 500, 500))
        self.plot_cooling.setBackground('w')

        self.__pen = pg.mkPen(color='b', width=5)

        self.plot_cooling.getAxis('left').setPen('black')
        self.plot_cooling.getAxis('left').setTextPen('black')
        self.plot_cooling.getAxis('bottom').setPen('black')
        self.plot_cooling.getAxis('bottom').setTextPen('black')

        self.plot_cooling.setTitle('<span style="color: black; font-size: 18px">Temperature</span>')
        self.plot_cooling.setLabel("left", '<span style="color: black; font-size: 18px">Temperature</span>')
        self.plot_cooling.setLabel("bottom", '<span style="color: black; font-size: 18px">Iteration</span>')

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

        self.b_cooling.setText(QtCore.QCoreApplication.translate("MainWindow", u"Plot cooling", None))
        self.b_cooling.clicked.connect(self.__plot_clicked)

        self.browse_button.setText(QtCore.QCoreApplication.translate("MainWindow", u"Browse file", None))
        self.browse_button.clicked.connect(self.__open_dialog_box)

        self.run_button.setText(QtCore.QCoreApplication.translate("MainWindow", u"Run algorithm", None))
        self.run_button.clicked.connect(self.__open_result_window)


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
                cool_type = list(Cooling.CoolingTypes)[self.cool_types_box.currentIndex()]
                poly = self.poly_line.text()
                if poly == '':
                    poly = None
                else:
                    poly = int(self.poly_line.text())
                cool_f = Cooling.cooling_factory(float(self.init_temp_line.text()), float(self.final_temp_line.text()),
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
        if filename[0].endswith('.json'):
            self.__data_path = filename[0]
            msg.setText("Data loaded")
        else:
            msg.setWindowTitle("Error")
            msg.setText("Wrong file extension\nExpected .json")
            msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)

        msg.exec()

    def __open_result_window(self):
        if all(txt != '' for txt in [self.init_temp_line.text(), self.final_temp_line.text(), self.max_iter_line.text(),
                                     self.iters_line.text()]):
            if self.cool_types_box.currentText() == "Polynominal" and self.poly_line == '':
                pass
            else:
                init_temp = float(self.init_temp_line.text())
                final_temp = float(self.final_temp_line.text())
                max_iter = int(self.max_iter_line.text())
                cool_type = list(Cooling.CoolingTypes)[self.cool_types_box.currentIndex()]
                iters = int(self.iters_line.text())
                poly = self.poly_line.text()
                if poly == '':
                    poly = None
                else:
                    poly = int(self.poly_line.text())
                cool_f = Cooling.cooling_factory(init_temp, final_temp, max_iter, cool_type, iters, poly)

                self.window = QtWidgets.QMainWindow()
                self.ui = UiResultWindow(self.__data_path, cool_f, init_temp, max_iter)
                self.ui.setupUi(self.window)
                self.window.show()


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