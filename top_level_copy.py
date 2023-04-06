# if '__main__' == __name__:
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

# lead planetary scale height, density at base of atm, and CNO abundances
parameters = np.loadtxt('../../param.txt')[-1]
# check if there are multiple values for height/dense/abund, if not take the one row from param.txt
try:
    lenp = len(parameters)
    multvals = True
    parameters = np.loadtxt('../../param.txt')
    indices = range(len(parameters))
except:
    # print('failed, retry')
    parameters = np.loadtxt('../../param.txt')
    multvals = False
    indices = [0]

for index in indices:
    if multvals:
        params = parameters[index]
    else:
        params = parameters
    height = float(params[0])
    dense = 1e10 * float(params[1])
    abund = float(params[2])
    r_star = float(params[3])
    npt = int(params[4])
    # npt = int(params[4])
    #print(type(abund))
    print('grid point:', height, dense/1e10, abund, r_star, npt)
    
    # make directories for saving, define some constants
    initiate(height, dense, abund)
    
    tstart = time.time()
    vis = False
    nx, ny = 512, 512 #grid pts (nx by ny)
    rfunct = 'exp' # exponential drop-off of radial intensity for path projection
    useaia = True
    # npt = 60 #120 #201
    # nobs = 10 #20
    
    # make star, T1 and T2 are corona temps
    sta = Star(rstar=r_star, nx=nx, ny=ny, coneang=89.99, corkT1=0.24441407345734878, corkT2=0.6312271833718255)
    print('star done')
    # make planet
    pla = Planet(parent=sta, inclin=89, sclht=height, nbase=dense)
    print('planet done')
    print(pla.sclht)
    
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
    
    print('scale height = {}'.format(pla.sclht))
    lcave_cts, lcfull, lcsummed_src, lcsummed_cts, tgrid, do_chi = calcs(sta, pla, vis=True, useaia=useaia, twoComponent=1, anim=False, conecc=sta.conecc, rfunct=rfunct,
                            npt=npt, nobs=20, naia=60)
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
        gridpts = pd.read_csv('../../gridpts.csv')
        gridpt = np.array([height,dense/1e10,abund,r_star,npt])
        # check if gridpoint already in file
        if not (gridpts == gridpt).all(1).any(): 
            gridpt = pd.DataFrame(gridpt.reshape(1,-1), columns=list(gridpts))
            # gridpts = pd.concat([gridpts,gridpt_frame], ignore_index=True)
            gridpt.to_csv('../../gridpts.csv', mode='a', index=False, header=False)
        else:
            print('gridpt already found')
    except:
        print('gridpt not found')
        gridpts = pd.DataFrame(data = {'sclht':[height], 'dense':[dense/1e10], 'abund':[abund], 'rstar':[r_star], 'npt':[npt]})
        gridpts.to_csv('../../gridpts.csv', mode='w', index=False)
