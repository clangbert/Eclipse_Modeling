#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 20:59:38 2023

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
import os

def save_image(fig,filename,dpi=50):
    # with PdfPages(filename, dpi=dpi) as pdf:
    #     pdf.savefig(fig)
    # p = PdfPages(filename)
    plt.savefig(filename, format='pdf', dpi=dpi, bbox_inches='tight') 
    # p.close()

def lc_to_pdf(sclht, dense, abund, rad_sta, source='BROAD_SRC', image='AVG', inclin=89, naia=60, npt=120, directory='/Users/chaucerlangbert/Data/exoXtransit/flipbook3/', files=[], filnam_counter=None, graph=False, save=False, counter=None, old=False): 
    """ 
    Converts light curves to individual pdfs.
    
    Parameters
    ----------
    sclht : float
        Planetary scale height of atmosphere (Rjup).
    dense : float
        Density at exobase of planetary atmosphere (1e10 cm^-3).
    abund : float
        Planet CNO abundance relative to stellar.
    rad_sta : float
        Stellar radius (R_sun).
    source : string, optional
        Wavelengths of simulation lc. The default is 'BROAD_SRC'.
    image : string, optional
        Light curve, individual (AIA1-60) or averaged. The default is 'AVG'.
    inclin : float, optional
        Inclination of planetary orbit. The default is 89.
    naia : int, optional
        Number of AIA images used for light curve. The default is 60.
    npt : int, optional
        Number of points sampled for lc. The default is 120.
    files : list, optional
        List of filenames of light curves. The default is [].
    filnam_counter : int, optional
        Keeps track of which file # so as not to confuse the naming of pdfs. The default is None.
    graph : Boolean, optional
        If True, shows fitted light curve and allows saving of pdfs if wanted. The default is False.
    old : TYPE, optional
        From the olden days when file names were slightly different. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    sclht = float(sclht)
    dense = float(dense)
    abund = float(abund)
    rad_sta = float(rad_sta)
    npt = int(npt)
    inclin = float(inclin)
    
    # try:
    m_ind = None
    m_arr = pd.read_csv('mass_per_vals.csv')
    for ind, rad in enumerate(m_arr['radius']):
        if rad == rad_sta:
            m_ind = ind
    m_star = m_arr['mass'][m_ind]
    per = m_arr['period'][m_ind]
    # except:
    #     print('radius value not found in mass values table, using default mass=0.823 M_sun and per=2.21857312')
    #     m_star = 0.823
    #     per = 2.21857312

    # print(filnam_counter, files)
    
    if not old: # inclin != 89
        # filename = f'../../exoXtransit/eclipse_panes/data/sclht{sclht}/dense{dense}/abund{abund}_srad{rad_sta}_naia{naia}_npt{npt}_inclin{inclin}_corkT10.2444_corkT20.6312.fits'.format(
        #     sclht=sclht, dense=dense, abund=abund, rad_sta=rad_sta, naia=naia, npt=npt, inclin=inclin)
        filename = f'../../exoXtransit/eclipse_panes/data/sclht{sclht}/dense{dense}/abund{abund}_srad{rad_sta}_mrad{m_star}_naia{naia}_npt{npt}_inclin{inclin}_corkT10.2444_corkT20.6312.fits'.format(
            sclht=sclht, dense=dense, abund=abund, rad_sta=rad_sta, m_star=m_star, naia=naia, npt=npt, inclin=inclin)
        title = f'{source} {image}, sclht = {sclht}, dense = {dense}, abund = {abund}, rstar = {rad_sta}, inclin = {inclin}'.format(
            source, image, sclht, dense, abund, rad_sta, inclin)
    else:
        filename = f'../../exoXtransit/eclipse_panes/data/sclht{sclht}/dense{dense}/abund{abund}_srad{rad_sta}_naia{naia}_npt{npt}_corkT10.2444_corkT20.6312.fits'.format(
            sclht=sclht, dense=dense, abund=abund, rad_sta=rad_sta, naia=naia, npt=npt)
        title = f'{source} {image}, sclht = {sclht}, dense = {dense}, abund = {abund}, rstar = {rad_sta}'.format(
            source, image, sclht, dense, abund, rad_sta)
    
    try :
        os.mkdir(directory)
    except:
        holding = 1
    
    hdul = fits.open(filename) # make header data unit, which is a header and data array
    src = hdul[source] 
    # hdr = broad_src.header
    data = src.data
    phase = data.field('PHASE')
    period = 2.21857312 * 86400
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
    
    # print(width, depth, peak)
    
    pdf_name = '{}flipbook{}.pdf'.format(directory, filnam_counter)
    
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
        
        if filnam_counter != None:
            if save:
                save_image(fig, filename=pdf_name)
            files.append(pdf_name)
            filnam_counter += 1
            # print(filnam_counter)
        
        plt.show()
        
        
    # file_name = 'flipbook{}'.format(filnam_counter)
    # # print(file_name)
    # files.append(pdf_name)
    
    return filnam_counter, files, directory



def flipbook(pdfs, directory='/Users/chaucerlangbert/Data/exoXtransit/flipbook_v3/', name='flipbook_inc89.0.pdf'):
    """
    Makes a flipbook pdf from individual pdfs.

    Parameters
    ----------
    pdfs : list
        list of pdf file names
    flipbook_nam : str, optional
        Where/under which name to store the flipbook. The default is '/Users/chaucerlangbert/Data/exoXtransit/flipbook3/flipbook_v3.pdf'.

    Returns
    -------
    None.

    """
    merger = PdfMerger()
    
    for pdf in pdfs:
        merger.append(pdf)
    
    flipbook_nam = '{}{}'.format(directory, name)
    merger.write(flipbook_nam)
    merger.close()
    
    

counter = 0
new_files = []

abundance = 10.0
inclin = 89.5
for sclht in [0.1,0.2,0.3,0.4]:
    for dense in [1.0,10.0,100.0,1000.0]:
        for rad_sta in [0.1,0.3,0.588,0.788,1.0]:
            # print(sclht,dense,abundance,rad_sta)
            try:
                counter, new_files, directory = lc_to_pdf(sclht=sclht, dense=dense, abund=abundance, rad_sta=rad_sta, inclin=inclin, graph=True, filnam_counter=counter, files=new_files, counter=True, save=True, directory='/Users/chaucerlangbert/Data/exoXtransit/flipbook_89.5/')
            # new_files.append(file)
            except:
                print('file not found')

# print(new_files)
print('make flipbook')

flipbook(pdfs=new_files, directory=directory, name='flipbook_inc89.5.pdf')

print('done?')

