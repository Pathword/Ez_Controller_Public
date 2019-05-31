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


"""
step_plotter takes a transfer function and a max time variable and displays the step response, max velocity, and 
max acceleration. Returns a step info dictionary called info. info is later called from GUI to display into step
info table 
"""


# root function is the plotter function
def step_plotter(G_s,D_s, max_t,cssv=-1):

    T_s = ["(" + G_s[0] + ')*(' + D_s[0] + ")", "(" + G_s[1] + ')*(' + D_s[1] + ")"]

    if cssv != -1:
        s = 0
        #getting current ssv, lim s->0 of T_s
        ssv = eval(T_s[0])/eval(T_s[1])

        k = cssv/ssv

        #applying k to G_s[0], doesnt matter
        G_s[0] = str(k)+"*"+G_s[0]
        transfer = sym2transfer(G_s,D_s)

    else:

        transfer = sym2transfer(G_s,D_s)

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


G_s = ["1","1"]
D_s = ["(240*s)+400","(s**4)+(12*(s**3))+(72*(s**2))+(240*s)+400"]

max_t = 5
T = sym2transfer(G_s, D_s)
step_plotter(G_s,D_s,max_t,6)
plt.show()
