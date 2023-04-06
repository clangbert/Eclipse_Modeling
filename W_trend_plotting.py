#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 01:43:43 2023

@author: chaucerlangbert
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotting(filnam, var, param):
    data = pd.read_csv(filnam)
    variable = data[var]
    parameter = data[param]
    plt.scatter(variable, parameter)
    # plt.title(var + ' vs ' + param)
    plt.xlabel(var)
    plt.ylabel(param)
    plt.show()
    
# def 2D_table(filnam, var, param):
#     data = pd.read_csv(filnam)
#     variable = data[var]
#     parameter = data[param]
#     # plt.scatter(variable, parameter)
#     # plt.title(var + ' vs ' + param)
#     # plt.xlabel(var)
#     # plt.ylabel(param)
#     # plt.show()

# filnam = '../../W_points.csv'

# for sclht in [0.3,0.7,1.0]:
#     for dense in [1.0,3.0,10.0]:
#         for abund in [0.5,1.0,1.5]:
#             for rstar in [0.3,0.788,1.0]:
#                 print(sclht,dense,abund,rad_sta)
#                 plotting(filnam, var1, var2, sclht, dense, abund, rstar, naia = 60, npt = 120)

filnam = '../../W_points2.csv'

for var in ['sclht', 'dense', 'abund', 'rstar', 'inclin']:
    for param in ['width', 'depth', 'peak']:
        plotting(filnam=filnam, var=var, param=param)

# filnam = '../../W_points2.csv'