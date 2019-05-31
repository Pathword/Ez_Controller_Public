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






def step_anim(G_s, D_s, lb, ub, samples, max_t, nfps,cssv=-1):
    # takes x as variable, x value passed by list comprehension
    # pass max_y to set constant y limit
    def plot_anim(G_s, D_s, x, max_y, max_t,cssv=-1):

        # iterative plant and controller
        G_s_n = ["", ""]
        D_s_n = ["", ""]

        # simply replacing x with value, stored as strings.
        G_s_n[0] = G_s[0].replace("x", str(x))
        G_s_n[1] = G_s[1].replace("x", str(x))

        D_s_n[0] = D_s[0].replace("x", str(x))
        D_s_n[1] = D_s[1].replace("x", str(x))

        T_s_n = ["(" + G_s_n[0] + ')*(' + D_s_n[0] + ")", "(" + G_s_n[1] + ')*(' + D_s_n[1] + ")"]

        if cssv != -1:
            s = 0
            # getting current ssv, lim s->0 of T_s
            ssv = eval(T_s_n[0]) / eval(T_s_n[1])

            k = cssv / ssv

            # applying k to G_s[0], doesnt matter
            G_s_n[0] = str(k) + "*" + G_s_n[0]
            transfer = sym2transfer(G_s_n, D_s_n)

        else:

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
    def get_ylim(G_s, D_s, lb, ub, samples,cssv=-1):
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

            T_s_n = ["(" + G_s_n[0] + ')*(' + D_s_n[0] + ")", "(" + G_s_n[1] + ')*(' + D_s_n[1] + ")"]

            if cssv != -1:
                s = 0
                # getting current ssv, lim s->0 of T_s
                ssv = eval(T_s_n[0]) / eval(T_s_n[1])

                k = cssv / ssv

                # applying k to G_s[0], doesnt matter
                G_s_n[0] = str(k) + "*" + G_s_n[0]
                transfer = sym2transfer(G_s_n, D_s_n)

            else:

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
    max_y = get_ylim(G_s, D_s, lb, ub, samples,cssv)

    # getting root dir
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # testing if gifs folder exists, else make one
    if os.path.exists(ROOT_DIR + "\\gifs") == False:
        os.mkdir(ROOT_DIR + "\\gifs")

    # big boi, getting a list of images using list comp
    imageio.mimsave((ROOT_DIR + '\\gifs\\animated.gif'),
                    [plot_anim(G_s, D_s, x, max_y, max_t,cssv) for x in np.arange(lb, ub, ((ub - lb) / samples))], fps=nfps)

    # closing all figures
    plt.close("all")


"""
crit_delta plots a static image of how Os,Ts,max_v, and max_a, the major criteria, change over each value of x. 
Used with animated step response, can achieve desired behavior and criteria with animated step response 
"""


def crit_delta(G_s, D_s, lb, ub, samples, max_t,cssv=-1):
    # OS, Ts, Max V, Max A.

    def iterate(G_s, D_s, x, max_t,cssv=-1):
        # iterative plant and controller
        G_s_n = ["", ""]
        D_s_n = ["", ""]

        # simply replacing x with value, stored as strings.
        G_s_n[0] = G_s[0].replace("x", str(x))
        G_s_n[1] = G_s[1].replace("x", str(x))

        D_s_n[0] = D_s[0].replace("x", str(x))
        D_s_n[1] = D_s[1].replace("x", str(x))

        T_s_n = ["(" + G_s_n[0] + ')*(' + D_s_n[0] + ")", "(" + G_s_n[1] + ')*(' + D_s_n[1] + ")"]

        if cssv != -1:
            s = 0
            # getting current ssv, lim s->0 of T_s
            ssv = eval(T_s_n[0]) / eval(T_s_n[1])

            k = cssv / ssv

            # applying k to G_s[0], doesnt matter
            G_s_n[0] = str(k) + "*" + G_s_n[0]
            transfer = sym2transfer(G_s_n, D_s_n)

        else:

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
        info = iterate(G_s, D_s, x, max_t,cssv)
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




G_s = ["(s+10)","(s+x)"]
D_s = ["(240*s)+400","(s**4)+(12*(s**3))+(72*(s**2))+(240*s)+400"]

lb = 2
ub= 4
samples = 30
fps = 6
max_t = 7
cssv = 2

step_anim(G_s, D_s, lb, ub, samples, max_t, fps,cssv)
crit_delta(G_s, D_s, lb, ub, samples, max_t,cssv)
