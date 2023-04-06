#from initiate import *
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import scipy.interpolate as inter
import pandas as pd
from celluloid import Camera
from sherpa.astro.ui import *
#from bodies import *
#from transit_v3_strip import *
#from specmod_cts import *
from matplotlib.colors import LogNorm
from pycrates import read_file


tab = read_file('merged_ltc_gehrels.fits')
pha = tab.get_column('phase').values-0.5
rate = tab.get_column('counts').values/(.005*2.21857312*86400*5)

erate = np.sqrt(rate)/((pha[-1]-pha[0])*.005*2.21857312*86400*5)
print(rate, erate)

optran = pd.read_csv('optical transit.csv') #cgl had to convert .txt to csv in table2.ipynb
HJDref = np.min(optran['HJD'])
Gphase = (((optran['HJD'])-HJDref)/2.21857312)+0.5
phase = 0.5 + Gphase - np.array([int(i) for i in Gphase])
sortphase = np.sort(phase)
pphase = phase-sortphase[int(len(sortphase)/2)]+0.4985
window = np.maximum((pphase < 0.545), (pphase > pha[0]))
pphase = pphase[window]
fig,ax = plt.subplots(figsize=(12 ,6))
ax2 = ax.twinx()
oot = np.maximum((pha < 0.485), (pha > 0.515))
ootcts = np.mean(rate[oot])
ax2.step(pha, rate/ootcts, c='red', where='mid', marker='.', mfc='black', label='Stacked Chandra data', mec='black')
ax2.scatter(pphase, optran['RFlux'][window], label=('Optical Transit from [7]'), s=0.5) 
ax2.errorbar(pha, rate/ootcts, erate/ootcts, c='black', ls='')
ax.set_xlabel('Orbital phase (0.5 is midtransit)')
ax.set_ylabel('Counts/second')
ax2.plot([0.485, 0.485], [min((rate-erate)/ootcts), max((rate+erate)/ootcts)], c='grey', ls='--', alpha=0.5)
ax2.plot([0.515, 0.515], [min((rate-erate)/ootcts), max((rate+erate)/ootcts)], c='grey', ls='--', alpha=0.75, label='In-Transit Window')
ax2.legend(loc='lower left')
ax2.set_ylabel('Normalised Count Rate')
ax.set_xlim((0.455, 0.545))
fig.tight_layout()
fig.savefig('plots/chandra stacked.png', transparent=True)
fig.show()
