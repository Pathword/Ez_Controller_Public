"""
This program is the root app, initializes UI and sends requests to other programs
"""

# native dependenices
import sys
import os
import decorator

# UI dependencies
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

# math dependencies
import control as c
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymbolic as pmbl

# encoder dependencies
import imageio

global G_s, D_s

G_s = ['', '']
D_s = ['', '']

"""
class mygui is the gui class. Contains information that is relayed to the backend, all functions/scripts are called from mygui. 
"""


class mygui(QDialog):

    ###CUSTOM GUI OPTIONS

    # toggle animation options
    def toggle_anim_options(self, bool):
        if bool == 1:
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
        if bool == 0:
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

    # init gui
    def __init__(self):
        super(mygui, self).__init__()
        loadUi('app_cw.ui', self)
        self.setWindowTitle('Ez_Controller')
        self.pushButton.clicked.connect(self.on_pushButton_clicked)
        self.comboBox.activated.connect(self.pass_Net_Adap)

        # entries
        self.entries = 0

        # supressing animation options
        self.toggle_anim_options(0)

    """
    RUNNERS, execution types 
    """

    # step response
    def step_response(self, G_s, D_s, max_t):
        plt.close()

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

        # plt plot should be init, showing
        plt.show()

    # rlocus
    def root_locus(self, G_s, D_s):
        plt.close()

        # combine and convert to Transfer matrix
        T = sym2transfer(G_s, D_s)

        # display governing transfer function
        self.label_18.setText(str(T))
        self.label_18.setAlignment(Qt.AlignCenter)

        # execute function
        rootlocus_plotter(T)

    # bode
    def bode_plot(self, G_s, D_s):
        plt.close()

        # combine and convert to Transfer matrix
        T = sym2transfer(G_s, D_s)

        # display governing transfer function
        self.label_18.setText(str(T))
        self.label_18.setAlignment(Qt.AlignCenter)

        # execute function
        bode_plotter(T)

    # animated step response
    def animated_step_response(self, G_s, D_s, lb, ub, samples, max_t, fps):
        plt.close()

        # execute function
        step_anim(G_s, D_s, lb, ub, samples, max_t, fps)

        # telling user gif location
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        location = str((ROOT_DIR + "\\gifs\\animated.gif"))

        self.lineEdit_gifpath.setText(location)

        # encapsulation with double strings, os freaks out sometimes
        location = "\"" + location + "\""

        # displaying to user
        os.system(location)

    # criteria_vs_x, goes with animated step response
    def criteria_vs_x(self, G_s, D_s, lb, ub, samples, max_t):
        crit_delta(G_s, D_s, lb, ub, samples, max_t)


    #animated root locus
    def animated_root_locus(self,G_s,D_s,lb,ub,samples,fps):
        plt.close()
        rootlocus_anim(G_s,D_s,lb,ub,samples,fps)

        # telling user gif location
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        location = str((ROOT_DIR + "\\gifs\\animated_rootlocus.gif"))

        self.lineEdit_gifpath.setText(location)

        # encapsulation with double strings, os freaks out sometimes
        location = "\"" + location + "\""

        # displaying to user
        os.system(location)

    """
    Run center, on run button push 
    """

    @pyqtSlot()
    def on_pushButton_clicked(self):
        self.entries += 1
        # pushbutton will send double entries for press/release, simply taking odd values for press
        if (self.entries % 2) == 1:

            # get mode
            mode = self.comboBox.currentText()

            # get line edits
            G_s[0] = self.lineEdit_1.text()
            G_s[1] = self.lineEdit_2.text()
            D_s[0] = self.lineEdit_3.text()
            D_s[1] = self.lineEdit_4.text()

            # get max T value
            max_t = int(self.spinBox_1.value())

            # error message box
            def error_msg(message):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText(message)
                msg.setWindowTitle("Error")
                msg.exec_()

            """
            cleaning center

            ^   ==  **
            )(  ==  )*(
            s(  ==  s*(
            )s  ==  )*s
            ns  ==  n*s
            sn  ==  s*n
            xn  ==  x*n
            nx  ==  n*x
            sx  ==  s*x
            xs  ==  x*s
            n() ==  n*()
            ()n ==  ()*n
            """

            # defaulting to 1 if no value, strings for sym2transfer to handle
            if G_s[0] == "":
                G_s[0] = "1"
            if G_s[1] == "":
                G_s[1] = "1"
            if D_s[0] == "":
                D_s[0] = "1"
            if D_s[1] == "":
                D_s[1] = "1"

            # defining syntaxers and replacements
            syntaxers = {'^': '**', ')(': ')*(', 's(': 's*(', ')s': ')*s', 'sx': 's*x', 'xs': 'x*s'}

            # making replacements for known syntaxes
            for key in syntaxers:
                G_s[0] = G_s[0].replace(key, syntaxers[key])
                G_s[1] = G_s[1].replace(key, syntaxers[key])
                D_s[0] = D_s[0].replace(key, syntaxers[key])
                D_s[1] = D_s[1].replace(key, syntaxers[key])

            # making replacements for "ns/sn", "nx/xn", value times s.
            for n in range(0, 9):
                # ns
                G_s[0] = G_s[0].replace(str(n) + "s", str(n) + "*s")
                G_s[1] = G_s[1].replace(str(n) + "s", str(n) + "*s")
                D_s[0] = D_s[0].replace(str(n) + "s", str(n) + "*s")
                D_s[1] = D_s[1].replace(str(n) + "s", str(n) + "*s")

                # sn
                G_s[0] = G_s[0].replace("s" + str(n), "s*" + str(n))
                G_s[1] = G_s[1].replace("s" + str(n), "s*" + str(n))
                D_s[0] = D_s[0].replace("s" + str(n), "s*" + str(n))
                D_s[1] = D_s[1].replace("s" + str(n), "s*" + str(n))

                # nx
                G_s[0] = G_s[0].replace(str(n) + "x", str(n) + "*x")
                G_s[1] = G_s[1].replace(str(n) + "x", str(n) + "*x")
                D_s[0] = D_s[0].replace(str(n) + "x", str(n) + "*x")
                D_s[1] = D_s[1].replace(str(n) + "x", str(n) + "*x")

                # xn
                G_s[0] = G_s[0].replace(str(n) + "x", str(n) + "*x")
                G_s[1] = G_s[1].replace(str(n) + "x", str(n) + "*x")
                D_s[0] = D_s[0].replace(str(n) + "x", str(n) + "*x")
                D_s[1] = D_s[1].replace(str(n) + "x", str(n) + "*x")

                # n()
                G_s[0] = G_s[0].replace(str(n) + "(", str(n) + "*(")
                G_s[1] = G_s[1].replace(str(n) + "(", str(n) + "*(")
                D_s[0] = D_s[0].replace(str(n) + "(", str(n) + "*(")
                D_s[1] = D_s[1].replace(str(n) + "(", str(n) + "*(")

                # ()n
                G_s[0] = G_s[0].replace(")" + str(n), ")*" + str(n))
                G_s[1] = G_s[1].replace(")" + str(n), ")*" + str(n))
                D_s[0] = D_s[0].replace(")" + str(n), ")*" + str(n))
                D_s[1] = D_s[1].replace(")" + str(n), ")*" + str(n))

            # check entry, used to check if x found somewhere
            check_entry = G_s[0] + G_s[1] + D_s[0] + D_s[1]

            """            
            MODE RUNS 
            """

            # mode: step response, add max T condition
            if mode == "Step Response":
                try:
                    self.step_response(G_s, D_s, max_t)
                except:
                    if "x" in check_entry:
                        error_msg("Variable \'x\' only to be used in Animated Step Response Mode")
                    else:
                        error_msg("There is either a syntax error or the system you have generated is unstable.")

            # mode: root locus
            if mode == "Root Locus":
                try:
                    self.root_locus(G_s, D_s)
                except:
                    if "x" in check_entry:
                        error_msg("Variable \'x\' only to be used in Animated Step Response Mode")
                    else:
                        error_msg("There is either a syntax error or the system you have generated is unstable.")

            # mode: Bode Plot
            if mode == "Bode Plot":
                try:
                    self.bode_plot(G_s, D_s)
                except:
                    if "x" in check_entry:
                        error_msg("Variable \'x\' only to be used in Animated Step Response Mode")
                    else:
                        error_msg("There is either a syntax error or the system you have generated is unstable.")

            # mode: animated step response
            if mode == "Animated Step Response":
                # get other info besides G_s,D_s
                try:
                    lb = float(self.lineEdit_lower.text())
                except:
                    error_msg("Please enter a value for lower bounds.")

                try:
                    ub = float(self.lineEdit_upper.text())
                except:
                    error_msg("Please enter a value for upper bounds")

                samples = int(self.spinBox_2.value())
                fps = int(self.spinBox_3.value())

                try:
                    if "x" not in check_entry:
                        # purposefully break by doing this, breaks try/except and passes to actual error message
                        # avoids double error message
                        e
                    else:
                        # animate
                        self.animated_step_response(G_s, D_s, lb, ub, samples, max_t, fps)
                        # plot criteria
                        self.criteria_vs_x(G_s, D_s, lb, ub, samples, max_t)

                except:
                    if "x" not in check_entry:
                        error_msg("Variable \'x\' not found")
                    else:
                        error_msg("There is either a syntax error or the system you have generated is unstable.")

            #mode: animated root locus
            if mode == "Animated Root Locus":
                # get other info besides G_s,D_s
                try:
                    lb = float(self.lineEdit_lower.text())
                except:
                    error_msg("Please enter a value for lower bounds.")

                try:
                    ub = float(self.lineEdit_upper.text())
                except:
                    error_msg("Please enter a value for upper bounds")

                samples = int(self.spinBox_2.value())
                fps = int(self.spinBox_3.value())

                try:
                    if "x" not in check_entry:
                        # purposefully break by doing this, breaks try/except and passes to actual error message
                        # avoids double error message
                        e
                    else:
                        # animate
                        self.animated_root_locus(G_s, D_s, lb, ub, samples, fps)

                except:
                    if "x" not in check_entry:
                        error_msg("Variable \'x\' not found")
                    else:
                        error_msg("There is either a syntax error or the system you have generated is unstable.")

    """
    GUI updater, toggle options 
    """

    # basically While True, getting mode changes
    @pyqtSlot()
    def pass_Net_Adap(self):

        if self.comboBox.currentText() == "Animated Step Response":
            self.toggle_anim_options(1)
        elif self.comboBox.currentText() == "Animated Root Locus":
            self.toggle_anim_options(1)
        else:
            self.toggle_anim_options(0)


"""
Action center, all actions called from gui. 
"""

"""
Sym2transfer converts G_s, D_s into a single T_s, then uses algebraic expansion to return a matrix with coefficients
on each order Availability up to 16th order s polynomial. Matrix is then inputted into get_T which calls from the 
controls library  
"""


def sym2transfer(G_s, D_s):
    # inputted as string
    def as_cof(expr):

        # string manipulation

        # if a negative is in there, pole/zero in RHP, unstable
        if "-" in expr:
            return "Unstable"

        # up to 16th order polynomial
        cof_mat = [0] * 16

        # getting rid of white spaces,
        expr_list = expr.replace(" ", "").split("+")
        # cof_list not organized, splitting be the cof and the n order
        # transfer functions like the cofs in reverse
        for n in expr_list:
            # any order greater than 1 represented with **, doing a check for s**1 and s**0 (s and simple cof)
            if "**" not in n:
                # if just s
                if n == "s":
                    cof_mat[-2] = 1
                # cof
                if "s" not in n:
                    cof_mat[-1] = float(n)

                # even if s passing for some reason, fixed
                elif len(n) > 1:
                    # to the left of the * sign
                    cof = float(n.split("*")[0])
                    cof_mat[-2] = float(cof)

            # checking for s**n where n>1
            if "**" in n:
                # adding a negative sign to order, reverse assigning in cof_mat. adding 1 space for negative indexing
                order = int("-" + n.split("**")[1]) - 1
                # cof always appears to the left of *, taking first instance
                # if no cof then first instance will be s, checking and defaulting to 1 if case
                first = n.split("*")[0]
                if first == "s":
                    cof_mat[order] = 1
                else:
                    cof_mat[order] = float(first)

        return cof_mat

    def get_T(G_s, D_s):
        # applying D_s to G_s, initializing resultant T_s
        T_s = ["(" + G_s[0] + ')*(' + D_s[0] + ")", "(" + G_s[1] + ')*(' + D_s[1] + ")"]
        # init T
        T = {}
        # init s symbol
        s = pmbl.var("s")

        # evaluating string :

        # finding isolated s powers, pymbolic sucks RIP sympy
        def get_iso_s(expr):
            s = pmbl.var("s")

            # checking if multiplication
            if "*" not in expr:
                return str(expr)

            # checking if power of s
            for x in range(2, 9):
                if expr == "s**" + str(x):
                    return expr

            # if s just return s
            if expr != "s":

                # coeff
                if "s" not in expr:
                    return str(expr)

                # get the children of the object, should be Variable("s")
                child_list = list(pmbl.expand(eval(expr)).children)

                # count s occ
                expr_list = [str(x) for x in child_list]

                # get order
                iso_s_order = expr_list.count('s')

                expr_list = [str(x) for x in expr_list if x != "s"]

                if iso_s_order > 0:
                    # if no coefficient example (s*s)
                    if iso_s_order == len(child_list):
                        expr = "s**" + str(iso_s_order)
                    # if coefficient example (2*s*s)
                    else:
                        expr = "*".join(expr_list) + "*(s**" + str(iso_s_order) + ")"

                expr = str(pmbl.expand(eval(expr)))
            return expr

        num_expr = pmbl.expand(eval(T_s[0]))
        den_expr = pmbl.expand(eval(T_s[1]))

        # expanding itself 3 times, JUST IN CASE, sometimes expanding once throws a fit
        for n in range(3):
            num_expr = pmbl.expand(num_expr)
            den_expr = pmbl.expand(den_expr)

        # expand to get isolated s
        num_expr = get_iso_s(str(num_expr))
        den_expr = get_iso_s(str(den_expr))

        # expanding one last time JUST IN CASE
        num_expr = pmbl.expand(eval(num_expr))
        den_expr = pmbl.expand(eval(den_expr))

        # get coefficients from expanded polynomial
        T['num'] = as_cof(str(num_expr))
        T['den'] = as_cof(str(den_expr))

        transfer = c.tf(T['num'], T['den'])
        return transfer

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
    OS = round(info['Overshoot'], 3)
    Ts = round(info['SettlingTime'], 3)
    SSv = round(info['SteadyStateValue'], 3)
    peak = round(info['Peak'], 3)
    max_v = round(max(dy), 4)
    max_a = round(max(ddy), 4)

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

    plt.title("OS: " + str(OS) + " | Ts: " + str(Ts) + '\n' + 'SSv: ' + str(SSv) + ' | Peak: ' + str(
        peak) + '\n' + "Max Velocity: " + str(max_v) + ' | Max Acceleration: ' + str(max_a))
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

    # plt should be init, showing within class structure of ui, update stepinfo fields first

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

        # maximize

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
crit_delta plots a static image of how Os,Ts,max_v, and max_a, the major criteria, change over each value of x. 
Used with animated step response, can achieve desired behavior and criteria with animated step response 
"""


def crit_delta(G_s, D_s, lb, ub, samples, max_t):
    # OS, Ts, Max V, Max A.

    def iterate(G_s, D_s, x, max_t):
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

        """
        decrease sample size? performance time? 
        """
        # defining time
        t = np.linspace(0, max_t, 1000)
        # returns numpy array
        sr = c.step_response(transfer, t)
        # parsing array
        y = sr[1]

        # getting step info
        info = c.step_info(transfer)
        # getting step info
        Os = info['Overshoot']
        Ts = info['SettlingTime']

        # getting velocity
        dy = []
        time_step = t[1] - t[0]
        for n in range(999):
            dy.append((y[n + 1] - y[n]) / time_step)

        max_v = max(np.abs(dy))

        # getting acceleration
        ddy = []
        for n in range(998):
            ddy.append((dy[n + 1] - dy[n]) / time_step)

        max_a = max(np.abs(ddy))

        return [Os, Ts, max_v, max_a]

    # init lists

    Os_list = []
    Ts_list = []
    max_v_list = []
    max_a_list = []

    # taking 0.01 from max_t, arange is non inclusive (<> not =<>=)
    x_range = np.arange(lb, ub - 0.01, float((ub - lb) / samples))

    # iterate and get values
    for x in x_range:
        info = iterate(G_s, D_s, x, max_t)
        Os_list.append(info[0])
        Ts_list.append(info[1])
        max_v_list.append(info[2])
        max_a_list.append(info[3])

    # getting absolute max
    abs_min_Os = round(min(np.abs(Os_list)), 2)
    abs_min_Ts = round(min(np.abs(Ts_list)), 3)
    abs_max_v = round(max(np.abs(max_v_list)), 3)
    abs_max_a = round(max(np.abs(max_a_list)), 3)

    # plot, title should include absolute max values

    # plotting OS
    plt.subplot(4, 1, 1)
    plt.title("Absolute Maximums" + "\n" + "Velocity: " + str(abs_max_v) + " | Acceleration: " + str(abs_max_a) + "\n" + "Absolute Minimums" + "\n" + "OS: " + str(abs_min_Os) + " | Ts: " + str(abs_min_Ts))

    plt.plot(x_range, Os_list, 'g')
    plt.ylabel("%OS")
    plt.grid(True)

    # plotting Ts
    plt.subplot(4, 1, 2)
    plt.plot(x_range, Ts_list, 'b')
    plt.ylabel("Settling Time")
    plt.grid(True)

    # plotting max_v
    plt.subplot(4, 1, 3)
    plt.plot(x_range, max_v_list, 'y')
    plt.ylabel("Max Velocity")
    plt.grid(True)

    # plotting Ts
    plt.subplot(4, 1, 4)
    plt.plot(x_range, max_a_list, 'r')
    plt.ylabel("Max Acceleration")
    plt.xlabel("x")
    plt.grid(True)

    plt.show()


"""
rootlocus animator 
"""

def rootlocus_anim(G_s, D_s, lb, ub, samples, nfps):
    plt.ioff()

    # takes x as variable, x value passed by list comprehension
    # pass max_y to set constant y limit
    def plot_anim(G_s, D_s, x):

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

        #plotting transfer
        fig,ax = plt.subplots()
        c.root_locus(transfer,grid=True,Plot=True,xlim=[-10,0],ylim=[-10,10])
        plt.title("x = " + str(x))
        fig = plt.gcf()

        # maximize

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()
        return image


    # getting root dir
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # testing if gifs folder exists, else make one
    if os.path.exists(ROOT_DIR + "\\gifs") == False:
        os.mkdir(ROOT_DIR + "\\gifs")

    # big boi, getting a list of images using list comp
    imageio.mimsave((ROOT_DIR + '\\gifs\\animated_rootlocus.gif'),
                    [plot_anim(G_s, D_s, x) for x in np.arange(lb, ub, ((ub - lb) / samples))], fps=nfps)

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
