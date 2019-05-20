"""
This program is the root app, initializes UI and sends requests to other programs

"""

# native dependenices
import sys
import os

# UI dependencies
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi

# math dependencies
import control as c
import matplotlib.pyplot as plt
import numpy as np
import sympy as spy

# encoder dependencies
import imageio

global G_s, D_s

G_s = ['', '']
D_s = ['', '']

"""
class mygui is the gui class. Contains information that is relayed to the backend, all functions/scripts are called from mygui. 
"""


class mygui(QDialog):


    def __init__(self):
        super(mygui, self).__init__()
        loadUi('app.ui', self)
        self.setWindowTitle('Ez_Controller')
        self.pushButton.clicked.connect(self.on_pushButton_clicked)
        self.comboBox.activated.connect(self.pass_Net_Adap)

        #entries
        self.entries = 0

        # default supressing animated options
        self.spinBox_2.setDisabled(1)
        self.spinBox_3.setDisabled(1)
        self.lineEdit_lower.setDisabled(1)
        self.lineEdit_upper.setDisabled(1)
        self.label_9.setDisabled(1)
        self.label_10.setDisabled(1)
        self.label_11.setDisabled(1)
        self.label_12.setDisabled(1)
        self.label_13.setDisabled(1)
        self.label_15.setDisabled(1)

    @pyqtSlot()
    def on_pushButton_clicked(self):
        self.entries += 1
        print(self.entries)
        # pushbutton will send double entries for press/release, simply taking odd values for press
        if (self.entries % 2) == 2:

            ### RUN CENTER

            # defining mode
            mode = self.comboBox.currentText()

            # get line edits
            G_s[0] = self.lineEdit_1.text()
            G_s[1] = self.lineEdit_2.text()
            D_s[0] = self.lineEdit_3.text()
            D_s[1] = self.lineEdit_4.text()

            # defaulting to 1 if no value, strings for sym2transfer to handle
            if G_s[0] == "":
                G_s[0] = "1"
            if G_s[1] == "":
                G_s[1] = "1"
            if D_s[0] == "":
                D_s[0] = "1"
            if D_s[1] == "":
                D_s[1] = "1"

            G_s[0] = G_s[0].replace("^", "**")
            G_s[1] = G_s[1].replace("^", "**")
            D_s[0] = D_s[0].replace("^", "**")
            D_s[1] = D_s[1].replace("^", "**")

            # get max T value
            max_t = int(self.spinBox_1.value())

            # mode: step response, add max T condition
            if mode == "Step Response":
                # combine and convert to Transfer matrix
                T = sym2transfer(G_s, D_s)

                # display governing transfer function
                self.label_18.setText(str(T))
                self.label_18.setAlignment(Qt.AlignCenter)

                # execute function and getting info, plot should show regardless
                info = step_plotter(T, max_t)

                # getting table info into a list, trust me better than handling dict.
                table_info = [0] * 9
                table_info[0] = info["Overshoot"]
                table_info[1] = info["SettlingTime"]
                table_info[2] = info["SteadyStateValue"]
                table_info[3] = info["Peak"]
                table_info[4] = info["MaxVelocity"]
                table_info[5] = info["MaxAcceleration"]
                table_info[6] = info["PeakTime"]
                table_info[7] = info["RiseTime"]
                table_info[8] = info["Undershoot"]

                # writing to Step info table in UI
                for n in range(0, 9):
                    self.tableWidget.setItem(0, n, QtWidgets.QTableWidgetItem(str(table_info[n])))

            # mode: root locus
            if mode == "Root Locus":
                # combine and convert to Transfer matrix
                T = sym2transfer(G_s, D_s)

                # display governing transfer function
                self.label_18.setText(str(T))
                self.label_18.setAlignment(Qt.AlignCenter)

                # execute function
                rootlocus_plotter(T)

            # mode: Bode Plot
            if mode == "Bode Plot":
                # combine and convert to Transfer matrix
                T = sym2transfer(G_s, D_s)

                # display governing transfer function
                self.label_18.setText(str(T))
                self.label_18.setAlignment(Qt.AlignCenter)

                # execute function
                bode_plotter(T)

            # mode: animated step response
            if mode == "Animated Step Response":
                lb = float(self.lineEdit_lower.text())
                ub = float(self.lineEdit_upper.text())
                samples = int(self.spinBox_2.value())
                fps = int(self.spinBox_3.value())

                # execute function
                step_anim(G_s, D_s, lb, ub, samples, max_t, fps)

                # telling user gif location
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                location = (ROOT_DIR + '\\gifs\\animated.gif')

                self.label_16.setText(location)

    
    # basically While True, getting mode changes
    @pyqtSlot()
    def pass_Net_Adap(self):

        if self.comboBox.currentText() == "Animated Step Response":
            # show animated options
            self.spinBox_2.setEnabled(1)
            self.spinBox_3.setEnabled(1)
            self.lineEdit_lower.setEnabled(1)
            self.lineEdit_upper.setEnabled(1)
            self.label_9.setEnabled(1)
            self.label_10.setEnabled(1)
            self.label_11.setEnabled(1)
            self.label_12.setEnabled(1)
            self.label_13.setEnabled(1)
            self.label_15.setEnabled(1)
        else:
            # hide animated options
            self.spinBox_2.setDisabled(1)
            self.spinBox_3.setDisabled(1)
            self.lineEdit_lower.setDisabled(1)
            self.lineEdit_upper.setDisabled(1)
            self.label_9.setDisabled(1)
            self.label_10.setDisabled(1)
            self.label_11.setDisabled(1)
            self.label_12.setDisabled(1)
            self.label_13.setDisabled(1)
            self.label_15.setDisabled(1)


"""
Sym2transfer converts G_s, D_s into a single T_s, then uses algebraic expansion to return a matrix with coefficients
on each order Availability up to 10th order s polynomial. Matrix is then inputted into get_T which calls from the 
controls library  
"""


def sym2transfer(G_s, D_s):
    def sym2mat(expr):
        if 's' not in expr:
            # coefficient
            return [float(eval(expr))]

        # expanding
        expanded = spy.expand(expr)
        # getting coefficients
        sym_mat = spy.Poly(expanded).coeffs()

        # getting expanded coefficients
        sym_dict = expanded.as_coefficients_dict()

        # adding zeros for lower order coeff

        num_dict = {}
        # converting current keys to strings
        for key in sym_dict.keys():
            num_dict.setdefault(str(key), sym_dict[key])

        # try declaring that value in the mat_dict, if it does exist great, if it doesnt = 0.
        mat_dict = {}
        # trying up to 10th order polynomial
        for n in range(0, 10):
            try:
                mat_dict[n] = (num_dict['s**' + str(n)])
            except:
                mat_dict[n] = 0

        # special cases of s and 1.
        try:
            mat_dict[1] = num_dict['s']
        except:
            mat_dict[1] = 0

        try:
            mat_dict[0] = num_dict['1']
        except:
            mat_dict[0] = 0

        # iterating backwards through mat_dict, appending values
        mat = []
        for n in range(9, -1, -1):
            mat.append(float(mat_dict[n]))

        return mat

    def get_T(G_s, D_s):
        # applying D_s to G_s, initializing resultant T_s
        T_s = ["(" + G_s[0] + ')*(' + D_s[0] + ")", "(" + G_s[1] + ')*(' + D_s[1] + ")"]
        # initializing T
        T = {}
        # converting:
        T['num'] = sym2mat(T_s[0])
        T['den'] = sym2mat(T_s[1])

        transfer = c.tf(T['num'], T['den'])
        return transfer

    # executing

    # init s symbol
    s = spy.Symbol('s')

    # execute
    T = get_T(G_s, D_s)

    return T


"""
step_plotter takes a transfer function and a max time variable and displays the step response, max velocity, and 
max acceleration. Returns a step info dictionary called info. info is later called from GUI to display into step
info table 

"""


# root function is the plotter function
def step_plotter(transfer, max_t):
    # defining time, 1000 discrete steps
    t = np.linspace(0, max_t, 1000)
    # returns numpy array
    sr = c.step_response(transfer, t)
    # parsing array
    y = sr[1]

    # getting velocity
    dy = []
    time_step = t[1] - t[0]
    for n in range(999):
        dy.append((y[n + 1] - y[n]) / time_step)

    # getting acceleration
    ddy = []
    for n in range(998):
        ddy.append((dy[n + 1] - dy[n]) / time_step)

    # getting step info
    info = c.step_info(transfer)
    OS = info['Overshoot']
    Ts = info['SettlingTime']
    SSv = info['SteadyStateValue']
    peak = info['Peak']

    # rounding so title is clean
    OS = round(OS, 3)
    Ts = round(Ts, 3)
    SSv = round(SSv, 3)
    peak = round(peak, 3)

    # plotting displacement over time to step response
    plt.subplot(2, 1, 1)
    plt.plot(t, y)
    plt.ylabel("Displacement")
    plt.xlim([0, max_t])

    # drawing Ts line
    plt.axvline(info['SettlingTime'], linewidth=1, linestyle='--', color='b')
    # drawing peak line
    plt.axhline(info['Peak'], linewidth=1, linestyle='--', color='g')
    # drawing SSv line
    plt.axhline(info['SteadyStateValue'], linewidth=1, linestyle='--', color='r')

    plt.title("OS: " + str(OS) + " | Ts: " + str(Ts) + '\n' + 'SSv: ' + str(SSv) + ' | Peak: ' + str(peak))
    plt.grid(True)

    # plotting velocity
    plt.subplot(4, 1, 3)
    plt.plot(t[:999], dy, "y")
    # drawing max v line
    plt.axhline(max(dy), linewidth=1, linestyle='--', color='g')
    plt.grid(True)
    plt.ylabel("Velocity")
    plt.xlim([0, max_t])

    # plotting acceleration
    plt.subplot(4, 1, 4)
    plt.plot(t[:998], ddy, 'r')
    plt.ylabel("Acceleration")
    # drawing max v line
    plt.axhline(max(ddy), linewidth=1, linestyle='--', color='g')
    plt.grid(True)
    plt.xlim([0, max_t])
    plt.xlabel('Time')

    plt.show()

    # rounding info dictionary
    for key in info.keys():
        info[key] = round(info[key], 4)

    # adding max v and max a
    info['MaxVelocity'] = round(max(dy), 4)
    info['MaxAcceleration'] = round(max(ddy), 4)

    return info


"""
Root locus plotter... takes transfer function and uses controls library to plot 
"""


def rootlocus_plotter(transfer):
    c.root_locus(transfer, grid=True, PrintGain=True, Plot=True)
    plt.show()


"""
Bode plotter... takes transfer function and uses controls library to plot 
"""


def bode_plotter(transfer):
    c.bode(transfer, Plot=True)
    plt.show()


"""
step_anim takes a the plant and controller as variables. Each iteration a new governing transfer function is found. 
The variable used to iterate each governing transfer function is x. Takes lower, upper bound, max_t, and 
samples as variables. Range of x values is found from these properties. The function get_ylim() 
is a crude iteration ran first to gather the estimated max peak of range. All plots are then plotted using 
plot_anim with a set max y height to prevent changing x and y limits. Gif is generated using imageio, 
saved to current directory and then gifs folder. (ROOT_DIR + \\gifs\\animated.gif)   

"""


def step_anim(G_s, D_s, lb, ub, samples, max_t, nfps):
    # takes x as variable, x value passed by list comprehension
    # pass max_y to set constant y limit
    def plot_anim(G_s, D_s, x, max_y, max_t):

        # iterative plant and controller
        G_s_n = ["", ""]
        D_s_n = ["", ""]

        # simply replacing x with value, stored as strings.
        G_s_n[0] = G_s[0].replace("x", str(x))
        G_s_n[1] = G_s[1].replace("x", str(x))

        D_s_n[0] = D_s[0].replace("x", str(x))
        D_s_n[1] = D_s[1].replace("x", str(x))

        # redefining transfer
        transfer = sym2transfer(G_s_n, D_s_n)

        # defining time
        t = np.linspace(0, max_t, 1000)
        # returns numpy array
        sr = c.step_response(transfer, t)
        # parsing array
        y = sr[1]

        # getting step info
        info = c.step_info(transfer)
        # getting step info
        OS = info['Overshoot']
        Ts = info['SettlingTime']
        SSv = info['SteadyStateValue']
        peak = info['Peak']

        # rounding so title is clean
        OS = round(OS, 4)
        Ts = round(Ts, 4)
        SSv = round(SSv, 4)
        peak = round(peak, 4)

        fig, ax = plt.subplots()
        ax.set_ylim(0, max_y)
        ax.plot(t, y)
        # drawing Ts line
        ax.axvline(info['SettlingTime'], linewidth=1, linestyle='--', color='b')
        # drawing peak line
        ax.axhline(info['Peak'], linewidth=1, linestyle='--', color='g')
        # drawing SSv line
        ax.axhline(info['SteadyStateValue'], linewidth=1, linestyle='--', color='r')
        plt.title("OS: " + str(OS) + " | Ts: " + str(Ts) + '\n' + 'SSv: ' + str(SSv) + ' | peak: ' + str(
            peak) + '\n' + "x: " + str(round(x, 4)))
        ax.grid()

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    # pre iterating, getting max val from a smaller sample set
    def get_ylim(G_s, D_s, lb, ub, samples):
        max_peaks = []

        # getting smaller samples to save on rendering time
        samples = samples / 6

        # try samples, append max_peaks from sample step info
        for x in np.arange(lb, ub, ((ub - lb) / samples)):
            # iterative plant and controller
            G_s_n = ["", ""]
            D_s_n = ["", ""]

            # simply replacing x with value, stored as strings.
            G_s_n[0] = G_s[0].replace("x", str(x))
            G_s_n[1] = G_s[1].replace("x", str(x))

            D_s_n[0] = D_s[0].replace("x", str(x))
            D_s_n[1] = D_s[1].replace("x", str(x))

            # redefining transfer
            transfer = sym2transfer(G_s_n, D_s_n)

            # get step info of transfer
            info = c.step_info(transfer)
            # get peak value
            peak = info['Peak']
            # append peak value
            max_peaks.append(peak)

        # y lim is the max of the maximum peaks returned,
        max_y = max(max_peaks)

        return max_y

    # getting y lim
    max_y = get_ylim(G_s, D_s, lb, ub, samples)

    # getting root dir
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # testing if gifs folder exists, else make one
    if os.path.exists(ROOT_DIR + "\\gifs") == False:
        os.mkdir(ROOT_DIR + "\\gifs")

    # big boi, getting a list of images using list comp
    imageio.mimsave((ROOT_DIR + '\\gifs\\animated.gif'),
                    [plot_anim(G_s, D_s, x, max_y, max_t) for x in np.arange(lb, ub, ((ub - lb) / samples))], fps=nfps)

    # closing all figures
    plt.close("all")


"""
Run app, initialize. exit sys if closed. 
"""


def runApp():
    app = QApplication(sys.argv)
    widget = mygui()
    widget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    runApp()

