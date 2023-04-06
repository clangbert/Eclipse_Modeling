#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 23:06:46 2023

@author: chaucerlangbert
"""

# scale height calcs

kB = 1.380658e-16 # Boltzmann constant (erg K^-1)
Grav = 6.672e-8 # Gravitational constant (cm^3/gm/s^2)
mjup = 1.899e+30 # jovian mass (gm)
rjup = 6.9911e+9 # jovian radius (cm)


def acc(mass=1.142, radius=1.138):
    acc = Grav * (mass * mjup) / ((radius * rjup)**2)
    return acc

def temp_calc(sclht, mmm=1.4723905802e-24, acc=acc()):
    T = (sclht * rjup) * mmm * acc / kB
    return T

