#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 21:57:48 2023

@author: chaucerlangbert
"""

from initiate import *
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import scipy.interpolate as inter
import pandas as pd
from celluloid import Camera
from sherpa.astro.ui import *
from bodies import *
from transit_v5 import *
from specmod_cts import *
from fits_gen import *
from mk_bands_spec import *
from fitting import *
import pandas as pd


def transit_lc(height, dense, abund, rstar=0.788, inclin=89.99, npt=120, camval=0, anim=False, lite=False): # mstar=0.823, per=2.21857312
    height = float(height)
    dense = 1e10 * float(dense)
    abund = float(abund)
    rstar = float(rstar)
    # mstar = float(mstar)
    npt = int(npt)
    inclin = float(inclin)
    # per = float(per)
    
    # get mass & period from table dependent upon radius, from Pecaut & Mamajek (2013, ApJS, 208, 9; http://adsabs.harvard.edu/abs/2013ApJS..208....9P) and K3
    # there is probably a more elegant way to do this
    try:
        m_ind = None
        m_arr = pd.read_csv('mass_per_vals.csv')
        for ind, rad in enumerate(m_arr['radius']):
            if rad == rstar:
                m_ind = ind
        mstar = m_arr['mass'][m_ind]
        per = m_arr['period'][m_ind]
        name = m_arr['name'][m_ind]
    except:
        print('radius value not found in mass values table, using default name, mass=0.823 M_sun and per=2.21857312')
        mstar = 0.823
        per = 2.21857312
        name = 'HD189733A'
    
    print('grid point:', height, dense/1e10, abund, rstar, mstar, inclin, name) # npt, per
    
    # make directories for saving, define some constants
    initiate(height, dense, abund)
    
    tstart = time.time()
    vis = False
    nx, ny = 512, 512 #grid pts (nx by ny)
    rfunct = 'exp' # exponential drop-off of radial intensity for path projection
    useaia = True # use AIA images
    
    # make star, T1 and T2 are corona temps
    sta = Star(objnam=name, nx=nx, ny=ny, rstar=rstar, mstar=mstar, per=per, coneang=89.99, corkT1=0.24441407345734878, corkT2=0.6312271833718255) 
    print('star done')
    # make planet
    pla = Planet(parent=sta, inclin=inclin, sclht=height, nbase=dense) 
    print('planet done')
    # print(pla.sclht)
    
    # semi-major axis from Kepler's 3rd law
    semiA = ((sta.per ** 2) * (sta.mass + pla.mass) / 0.0134) ** 0.33333
    # Get an xray image by assuming emissivity is constant over volume
    xra = [-(2. * semiA + sta.rad + pla.rad) * 1.2 / 2., (2. * semiA + sta.rad + pla.rad) * 1.2 / 2.]
    yra = xra
    
    sta.solimg1 = sta.get_star_img(sta.rad, sta.coronaht1, vis, xra, yra, nx, ny, sta.corkT1, rfunct, sta.conecc,
                                   useaia=useaia)
    sta.solimg2 = sta.get_star_img(sta.rad, sta.coronaht2, vis, xra, yra, nx, ny, sta.corkT2, rfunct, sta.conecc,
                                   useaia=useaia)
    sta.solimgarr = [sta.solimg1, sta.solimg2]
    
    print('star done, now planet')
    
    pla.set_atm_abund(abundO=abund)
    # pla.set_sclht()
    print('check grid pt: ', pla.sclhtRJ, pla.nbase/1e10, pla.abundO, sta.rad, sta.mass, round(pla.inclin*180/np.pi,1), sta.name)
    
    # print('scale height = {}'.format(pla.sclht))
    lcave_cts, lcfull, lcsummed_src, lcsummed_cts, tgrid, do_chi = calcs(sta, pla, vis=True, useaia=useaia, twoComponent=1, anim=anim, conecc=sta.conecc, rfunct=rfunct,
                            npt=npt, nobs=20, naia=60, camera_val=camval, lite=lite) 
    print('out ofcalcs')
    
    #print(np.shape(lcfull[0]), avgtest)
    #sampchi = simp_chi_calc(lcfull[0]*1, nam='no')
    print(np.shape(lcave_cts))
    if do_chi:
        avgchi = poissoncalc(np.array([tgrid, lcave_cts[0]*1]))#, nam='sclht={} eV, nbase={}e+11, oxygen abund={}'.format(height, dense, abund), verbose=0)
    
        try:
            chiarr = pd.read_csv('../../poisson_probs.csv')
            chiarr = chiarr.append({'sclht':height, 'dense':dense, 'abund':abund, 'prob':avgchi}, ignore_index=True)
        #print(chiarr)
        except:
            print('chiarr not found')
            chiarr = pd.DataFrame(data = {'sclht':[height], 'dense':[dense], 'abund':[abund], 'prob':[avgchi]})
        chiarr.to_csv('../../poisson_probs.csv', index=False)
    else:
       print('poisson already found')
       
    try:
        gridpts = pd.read_csv('../../gridpts3.csv')
        gridpt = np.array([pla.sclht,pla.nbase/1e10,pla.abundO,sta.rad,sta.mass,npt,inclin,sta.per])
        # check if gridpoint already in file
        if not (gridpts == gridpt).all(1).any(): 
            gridpt = pd.DataFrame(gridpt.reshape(1,-1), columns=list(gridpts))
            # gridpts = pd.concat([gridpts,gridpt_frame], ignore_index=True)
            gridpt.to_csv('../../gridpts3.csv', mode='a', index=False, header=False)
        else:
            print('gridpt already found')
    except:
        print('gridpt not found')
        gridpts = pd.DataFrame(data = {'sclht':[pla.sclht],'dense':[pla.nbase/1e10],'abund':[pla.abundO],'rstar':[sta.rad],'mstar':[sta.mass],'npt':[npt],'inclin':[pla.inclin],'per':[sta.per]})
        gridpts.to_csv('../../gridpts3.csv', mode='w', index=False)
        # gridpts.to_csv('../../gridpts.csv', mode='w', index=False)

# load planetary scale height, density at base of atm, CNO abundances, stellar radius, number of points in lc, and inclination
# for height in [0.1,0.2]:
#     for dense in [0.3,1.0,3.0]:
#         for inclin in [88,88.5,89.5]:
#             transit_lc(height=height, dense=dense, abund=3, r_star=0.1, inclin=inclin, npt=120)

# transit_lc(height=1.0, dense=1.0, abund=1.0, rstar=0.1, inclin=89.5, npt=120, camval=50, anim=True)
# transit_lc(height=1.0, dense=1.0, abund=1.5, rstar=0.1, inclin=89.5, npt=120, camval=50, anim=True)
# transit_lc(height=1.0, dense=1.0, abund=1.5, rstar=0.3, inclin=89.0, npt=120, camval=50, anim=True)
# transit_lc(height=1.0, dense=1.0, abund=0.5, rstar=1.0, inclin=89.0, npt=120, camval=50, anim=True)
### transit_lc(height=1.0, dense=1.0, abund=1.0, camval=50, anim=True)
# transit_lc(height=0.7, dense=3.0, abund=1.0, camval=50, anim=True)

# for inclins in [89.0, 89.5, 90.0]:
#     for heights in [0.1,0.2,0.3,0.4,0.5]:
#         for denses in [0.3,1.0,3.0]:
#             for rstars in [0.1,0.3,0.788,1.0]:
#                 transit_lc(height=heights, dense=denses, inclin=inclins, abund=1.0, rstar=rstars, camval=50, anim=True, lite=True)

# transit_lc(height=0.2, dense=1.0, inclin=89.5, abund=1.0, rstar=0.3, camval=50, anim=True, lite=True)
# transit_lc(height=0.3, dense=1.0, inclin=89.5, abund=1.0, rstar=0.3, camval=50, anim=True, lite=True)
# transit_lc(height=0.4, dense=1.0, inclin=89.5, abund=1.0, rstar=0.3, camval=50, anim=True, lite=True)

# transit_lc(height=0.4, dense=100, abund=10, inclin=89.99, camval=50, anim=True)
# transit_lc(height=0.3, dense=100, abund=10, inclin=89.99, camval=50, anim=True)

for inclin in [89.5, 87.0]:
    for sclht in np.arange(0.1,0.5,0.1):
        for dense in np.logspace(0,2,num=3,base=10):
            for rstar in [0.1,0.3,0.588,0.788,1.0]:
                transit_lc(height=sclht, dense=dense, abund=10, rstar=rstar, inclin=inclin, camval=50, anim=True)


# for inclins in [89.5,90.0]: 
#     for heights in [0.1,0.2,0.3,0.4,0.5]:
#         for denses in [0.3,1.0,3.0]:
#             for rstars in [0.1,0.3,0.588,1.0]:
#                 transit_lc(height=heights, dense=denses, inclin=inclins, abund=1.0, rstar=rstars, camval=50, anim=True, lite=True)
