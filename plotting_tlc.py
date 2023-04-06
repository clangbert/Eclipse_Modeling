#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:13:16 2022

@author: chaucerlangbert
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import astropy
from astropy.io import fits
import csv
from scipy.optimize import curve_fit
# from ciao_contrib.runtool import *
# from ciao_contrib.runtool import acis_process_events, dmstat, dmmerge
# import ciao_contrib.runtool
######import ciao_contrib.runtool as rt
# from pycrates import read_file

sclht = 0.4 # 0.3, 0.7 [R_jup]
dense = 100.0 # 1.0
abund = 10.0 # 1.0, 3.0
rad_sta = 1.0 # 0.788
mstar = 1.0 # 0.823
naia = 60
# nobs = 10
npt = 120 # 120
inclin = 89.99
old = False

# try:
#     m_ind = None
#     m_arr = pd.read_csv('mass_per_vals.csv')
#     for ind, rad in enumerate(m_arr['radius']):
#         if rad == rad_sta:
#             m_ind = ind
#     mstar = m_arr['mass'][m_ind]
#     per = m_arr['period'][m_ind]
# except:
#     print('radius value not found in mass values table, using default mass=0.823 M_sun and per=2.21857312')
#     mstar = 0.823
#     per = 2.21857312

source = 'BROAD_SRC' # BROAD_SRC, USOFT_SRC, SOFT_SRC, MED_SRC, HARD_SRC
image = 'AVG' #AIA2 #AVG

if not old:
    filename = f'../../exoXtransit/eclipse_panes/data/sclht{sclht}/dense{dense}/abund{abund}_srad{rad_sta}_mrad{mstar}_naia{naia}_npt{npt}_inclin{inclin}_corkT10.2444_corkT20.6312.fits'.format(
        sclht=sclht, dense=dense, abund=abund, rad_sta=rad_sta, m_star=mstar, naia=naia, npt=npt, inclin=inclin)
    # filename = f'../../exoXtransit/eclipse_panes/data/sclht{sclht}/dense{dense}/abund{abund}_srad{rad_sta}_mrad{m_star}_naia{naia}_npt{npt}_inclin{inclin}_corkT10.2444_corkT20.6312.fits'.format(
    #     sclht=sclht, dense=dense, abund=abund, rad_sta=rad_sta, naia=naia, npt=npt, inclin=inclin)
    title = f'{source} {image}, sclht = {sclht}, dense = {dense}, abund = {abund}, rstar = {rad_sta}, inclin = {inclin}'.format(
        source, image, sclht, dense, abund, rad_sta, inclin)
else:
    filename = f'../../exoXtransit/eclipse_panes/data/sclht{sclht}/dense{dense}/abund{abund}_srad{rad_sta}_naia{naia}_npt{npt}_corkT10.2444_corkT20.6312.fits'.format(
        sclht=sclht, dense=dense, abund=abund, rad_sta=rad_sta, naia=naia, npt=npt)
    title = f'{source} {image}, sclht = {sclht}, dense = {dense}, abund = {abund}, rstar = {rad_sta}'.format(
        source, image, sclht, dense, abund, rad_sta)

# dataname = f'../../exoXtransit/eclipse_panes/data/sclht{round(plane.sclhtRJ, 5)}/dense{plane.nbase/(1e+10)}/abund{plane.abundO}_naia{naia}_corkT1{round(star.corkT1, 4)}_corkT2{round(star.corkT2, 4)}.fits'

# data = astropy.io.fits.open(filename)
hdul = fits.open(filename) # makes header data unit, which is a header and data array
# primary = hdul[0]
# pha_grid = hdul[1]
# src = hdul[2] # broad_src
src = hdul[source] 
# broad_src = hdul['BROAD']
# hdr = broad_src.header
data = src.data
phase = data.field('PHASE')
period = 2.21857312 * 86400
times = period * phase # /
avg = data.field(image)
avg_norm = avg / np.max(avg) # normalize!
# avg_norm = (avg - np.min(avg)) / (np.max(avg) - np.min(avg)) # normalize!


# find bottom point(s) of W by interpolating and finding where derivatives are zero
fit = UnivariateSpline(phase, avg, k=4, s=0)
fit_norm = UnivariateSpline(phase, avg_norm, k=4, s=0)
zeroes = fit.derivative().roots()


# find elbows
elbow_val = np.max(avg) - 0.05 * (np.max(avg) - np.min(avg)) # subtract 5% of depth from max value to get 0.95
status = 'above'
elbows = []
more_phases = np.arange(min(phase), max(phase), 0.00001)

for pha in more_phases:
    if status == 'above' and fit(pha) < elbow_val:
        elbows.append(pha)
        status = 'below'
    elif status == 'below' and fit(pha) > elbow_val:
        elbows.append(pha)
        status = 'above'

# get rid of zeroes before or after the transit actually occurs
mask = []
for ind, pha in enumerate(zeroes):
    if pha < elbows[0] or pha > elbows[-1]:
        mask.append(ind)
if len(mask) > 0:
    zeroes = np.delete(zeroes, np.array(mask))

### parabola fitting ###
if_parabola = False

def parabola(x, A, B, C):
    y = A*(x-B)**2 + C
    return y

# print('check parabola')
if len(zeroes) >= 3: # check if >= three zeroes, then can fit parabola to middle of transit btw bottom pts
    # print('parabola?')
    if_parabola = True
    zero_vals = fit(zeroes)
    new_zero_vals, new_zero_pha = zip(*sorted(zip(zero_vals, zeroes)))
    bottoms = np.array([new_zero_pha[0], new_zero_pha[1]])

    stat = 'before'
    parabola_bound_indices = []
    print(parabola_bound_indices)
    
    for i, pha in enumerate(phase):
        if stat == 'before' and pha > bottoms[0]:
            parabola_bound_indices.append(i)
            stat = 'mid'
        elif stat == 'mid' and pha > bottoms[-1]:
            parabola_bound_indices.append(i-1)
            stat = 'after'
            
    print(parabola_bound_indices)
    try:
        parabola_phases = phase[parabola_bound_indices[0]:parabola_bound_indices[1]]
        parabola_vals = fit(parabola_phases)
    
        parameters, covariance = curve_fit(parabola, parabola_phases, parabola_vals)
        
        fit_A = parameters[0]
        fit_B = parameters[1]
        fit_C = parameters[2]
        fit_parabola = parabola(parabola_phases, fit_A, fit_B, fit_C)
        
        if fit_B > phase[parabola_bound_indices[0]] and fit_B < phase[parabola_bound_indices[1]]:
            B = fit_B
        elif fit_B < phase[parabola_bound_indices[0]]:
            B = phase[parabola_bound_indices[0]]
        else:
            B = phase[parabola_bound_indices[1]]
        
        parab_params = np.array([fit_A, B, fit_C])
    except:
        if_parabola = False
        bottoms = np.array([zeroes])
        parab_params = np.array([np.NaN, np.NaN, np.NaN])
else:
    bottoms = np.array([zeroes])
    parab_params = np.array([np.NaN, np.NaN, np.NaN])

W_points = np.append(elbows, bottoms)
W_points_norm = fit_norm(W_points)
if if_parabola == True:
    W_points_phases = np.append(W_points, B)
else:
    W_points_phases = W_points
# W_points_phases = np.sort(W_points)
W_points_values = fit(W_points_phases)
W_params = np.append(W_points, parab_params)

# plotting
fig, ax = plt.subplots()
ax2 = ax.twinx() # normalized count rate on right
ax3 = ax.twiny() # times [sec] up top

ax.plot(phase, avg, '.', label = 'Model')
ax2.plot(phase, avg_norm, '.', alpha = 0)
ax3.plot(times, avg, '.', alpha = 0)
ax.plot(more_phases, fit(more_phases), '-', label = 'Model Fit')
ax.plot(W_points_phases, W_points_values, 'o', color = 'red', zorder = 5, label = 'W Points')
if if_parabola == True:
    ax.plot(parabola_phases, parabola_vals, '.', label = 'Parabola')
    ax.plot(parabola_phases, fit_parabola, '-', label='Parabola Fit')
ax.legend()
ax.set_xlabel('Phase')
ax.set_ylabel('Counts per second')
ax.set_title(title)

# ax2.set_ylim((min(avg_norm),max(avg_norm)))
ax2.set_ylabel('Normalized Count Rate')
ax2.plot([],[])

# ax3.set_xlim(min(times),max(times))
ax3.set_xlabel('Time [sec]')
ax3.plot([],[])

plt.show()

# below should change to try: add points to already made file, except: make file and add points

# # write csv file with W points
# with open('W_points.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(["Sclht","Image","Phase_top_left", "Phase_top_right", "Phase_bottom_left", "Phase_bottom_right", "A", "B", "C", "Norm_tl", "Norm_tr", "Norm_bl", "Norm_br"])
#     sys_params = np.array([sclht, image])
#     row_phases = np.append(sys_params, W_params)
#     row_norms = np.array([W_points_norm])
#     row = np.append(row_phases, row_norms)
#     writer.writerow(row.tolist())
#     # writer.writerows(zip(W_points_phases, W_points_values))
