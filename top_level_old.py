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

params = np.loadtxt('../../exoXtransit/param.txt')[-1]
try:
    lenp = len(params)
except:
    #print('failed, retry')
    params = np.loadtxt('../../exoXtransit/param.txt')
height = float(params[0])
dense = 1e10 * float(params[1])
abund = float(params[2])
#print(type(abund))
#print(height, dense, abund)

initiate(height, dense, abund)

tstart = time.time()
vis = False
nx, ny = 512, 512
rfunct = 'exp'
useaia = True

sta = Star(nx=nx, ny=ny, coneang=89.99, corkT1=0.24441407345734878, corkT2=0.6312271833718255)
print('star done')
pla = Planet(parent=sta, inclin=89, sclht=height, nbase=dense)
print('planet done')
print(pla.sclht)

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
                        npt=201, naia=60)
print('out ofcalcs')

#print(np.shape(lcfull[0]), avgtest)
#sampchi = simp_chi_calc(lcfull[0]*1, nam='no')
print(np.shape(lcave_cts))
if do_chi:
    avgchi = poissoncalc(np.array([tgrid, lcave_cts[0]*1]))#, nam='sclht={} eV, nbase={}e+11, oxygen abund={}'.format(height, dense, abund), verbose=0)

    try:
        chiarr = pd.read_csv('../../exoXtransit/poisson_probs.csv')
        chiarr = chiarr.append({'sclht':height, 'dense':dense, 'abund':abund, 'prob':avgchi}, ignore_index=True)
    #print(chiarr)
    except:
        print('chiarr not found')
        chiarr = pd.DataFrame(data = {'sclht':[height], 'dense':[dense], 'abund':[abund], 'prob':[avgchi]})
    chiarr.to_csv('../../exoXtransit/poisson_probs.csv', index=False)
else:
   print('poisson already found')
