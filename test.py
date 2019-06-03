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
                    return str(expr)

            # if s just return s
            if expr == "s":
                return str(expr)
            else:
                # coeff
                if "s" not in expr:
                    return str(expr)

                # get the children of the object, should be Variable("s")
                child_list = list(pmbl.expand(eval(expr)).children)

                # count s occ
                expr_list = [str(x) for x in child_list]

                print(expr_list)
                # get order
                iso_s_order = expr_list.count('s')


                ###WHY IS THIS CONDITIONAL HERE?
                expr_list = [str(x) for x in expr_list]

                print(iso_s_order)

                if iso_s_order > 0:
                    # if no coefficient example (s*s)
                    if iso_s_order == len(child_list):
                        expr = "s**" + str(iso_s_order)
                    # if coefficient example (2*s*s)
                    else:
                        print("else")
                        expr = "*".join(expr_list) + "*(s**" + str(iso_s_order) + ")"

                print(expr)

                expr = pmbl.expand(eval(expr))

                return str(expr)



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




G_s = ["1","1"]
D_s = ["2","(s**2)+(0.999*s)+3"]

sym2transfer(G_s,D_s)
