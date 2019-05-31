
import control as c
import pymbolic as pmbl

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

            # checking if just power of s
            for x in range(2, 9):
                if expr == "s**" + str(x):
                    return str(expr)

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
step_anim takes a the plant and controller as variables. Each iteration a new governing transfer function is found. 
The variable used to iterate each governing transfer function is x. Takes lower, upper bound, max_t, and 
samples as variables. Range of x values is found from these properties. The function get_ylim() 
is a crude iteration ran first to gather the estimated max peak of range. All plots are then plotted using 
plot_anim with a set max y height to prevent changing x and y limits. Gif is generated using imageio, 
saved to current directory and then gifs folder. (ROOT_DIR + \\gifs\\animated.gif)   
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


G_s = ["(s+x)","(s+3)*(s+5)*(s+7)"]
D_s = ["1","1"]

rootlocus_anim(G_s,D_s,1,9,40,10)