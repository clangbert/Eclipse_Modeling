#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 20:11:42 2023

@author: chaucerlangbert
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import astropy
from astropy.io import fits
import csv
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger

def save_image(fig,filename,dpi=50):
    # with PdfPages(filename, dpi=dpi) as pdf:
    #     pdf.savefig(fig)
    # p = PdfPages(filename)
    plt.savefig(filename, format='pdf', dpi=dpi, bbox_inches='tight') 
    # p.close()

def W_points_calc(sclht, dense, abund, rad_sta, source='BROAD_SRC', image='AVG', inclin=89.0, per=2.21857312, naia=60, npt=120, graph=False, counter=None, save=False, old=False): 
    sclht = float(sclht)
    dense = float(dense)
    abund = float(abund)
    rad_sta = float(rad_sta)
    npt = int(npt)
    inclin = float(inclin)
    per = float(per)
    
    try:
        m_ind = None
        m_arr = pd.read_csv('mass_per_vals.csv')
        for ind, rad in enumerate(m_arr['radius']):
            if rad == r_star:
                m_ind = ind
        m_star = m_arr['mass'][m_ind]
        per = m_arr['period'][m_ind]
    except:
        print('radius value not found in mass values table, using default mass=0.823 M_sun and per=2.21857312')
        m_star = 0.823
        per = 2.21857312
    
    if not old: # inclin != 89
        filename = f'../../exoXtransit/eclipse_panes/data/sclht{sclht}/dense{dense}/abund{abund}_srad{rad_sta}_mrad{mstar}_naia{naia}_npt{npt}_inclin{inclin}_corkT10.2444_corkT20.6312.fits'.format(
            sclht=sclht, dense=dense, abund=abund, rad_sta=rad_sta, m_star=mstar, naia=naia, npt=npt, inclin=inclin)
        title = f'{source} {image}, sclht = {sclht}, dense = {dense}, abund = {abund},rstar = {rad_sta}, mstar = {m_star}, inclin = {inclin}'.format(
            source, image, sclht, dense, abund, rad_sta, inclin)
    else:
        filename = f'../../exoXtransit/eclipse_panes/data/sclht{sclht}/dense{dense}/abund{abund}_srad{rad_sta}_naia{naia}_npt{npt}_corkT10.2444_corkT20.6312.fits'.format(
            sclht=sclht, dense=dense, abund=abund, rad_sta=rad_sta, naia=naia, npt=npt)
        title = f'{source} {image}, sclht = {sclht}, dense = {dense}, abund = {abund}, rstar = {rad_sta}'.format(
            source, image, sclht, dense, abund, rad_sta)
    
    hdul = fits.open(filename) # make header data unit, which is a header and data array
    src = hdul[source] 
    # hdr = broad_src.header
    data = src.data
    phase = data.field('PHASE')
    period = per * 86400 # seconds
    times = period * phase 
    avg = data.field(image)
    avg_norm = avg / np.max(avg) # normalize!
    
    
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
    
    if len(zeroes) >= 3: # check if >= three zeroes, then can fit parabola to middle of transit btw bottom pts
        if_parabola = True
        zero_vals = fit(zeroes)
        new_zero_vals, new_zero_pha = zip(*sorted(zip(zero_vals, zeroes)))
        bottoms = np.array([new_zero_pha[0], new_zero_pha[1]])
    
        stat = 'before'
        parabola_bound_indices = []
        
        for i, pha in enumerate(phase):
            if stat == 'before' and pha > bottoms[0]:
                parabola_bound_indices.append(i)
                stat = 'mid'
            elif stat == 'mid' and pha > bottoms[-1]:
                parabola_bound_indices.append(i-1)
                stat = 'after'
    
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
    W_points_values = fit(W_points_phases)
    W_params = np.append(W_points, parab_params)
    
    # calculate width, depth, central peak
    W_times = period * np.array(elbows)
    
    width = W_times[1] - W_times[0]
    depth = min(fit_norm(bottoms))
    peak = parab_params[2] / np.max(avg)
    
    try:
        depth = float(depth)
    except:
        depth = depth[0]
    
    print(width, depth, peak)
    
    if graph:
        # plotting
        fig, ax = plt.subplots()
        ax2 = ax.twinx() # normalized count rate on right
        ax3 = ax.twiny() # times [sec] up top
        
        ax.plot(phase, avg, '.', label = 'Model')
        ax2.plot(phase, avg_norm, '.', alpha = 0)
        ax3.plot(times, avg, '.', alpha = 0)
        ax.plot(more_phases, fit(more_phases), '-', label = 'Model Fit')
        ax.plot(W_points_phases, W_points_values, 'o', color = 'red', zorder = 5, label = 'W Points')
        if if_parabola:
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
        
        if counter != None:
            filename = '/Users/chaucerlangbert/Data/exoXtransit/flipbook/flipbook{}.pdf'.format(filnam_counter)
            save_image(fig, filename=filename, dpi=dpi)
            files.append(filename)
        
        plt.show()
        
    
    if save:
        try:
            W_points = pd.read_csv('../../W_points3.csv')
            W_point = np.array([sclht,dense,abund,rad_sta,m_star,inc,npt,width,depth,peak])
            # check if gridpoint already in file
            if not (W_points == W_point).all(1).any(): 
                W_point = pd.DataFrame(W_point.reshape(1,-1), columns=list(W_points))
                W_point.to_csv('../../W_points3.csv', mode='a', index=False, header=False)
            else:
                print('W already found')
        except:
            print('W not found')
            W_points = pd.DataFrame(data = {'sclht':[sclht],'dense':[dense],'abund':[abund],'rstar':[rad_sta],'mstar':[m_star],'inc':[inc],'npt':[npt],'width':[width],'depth':[depth],'peak':[peak]})
            W_points.to_csv('../../W_points2.csv', mode='w', index=False)

filnam_counter = 0
files = []

for inc in [89.0, 89.5]:    
    for sclht in [0.1,0.2,0.3,0.4,0.5]:
        for dense in [0.3,1.0,3.0]:
            for abund in [1.0]:
                for rad_sta in [0.1,0.3,0.588,0.788,1.0]:
                    print(sclht,dense,abund,rad_sta,inc)
                    W_points_calc(sclht, dense, abund, rad_sta, inclin=inc, graph=True, save=True) # counter=filnam_counter
                    filnam_counter += 1






