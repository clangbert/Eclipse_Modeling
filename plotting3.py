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
#erate = tab.get_column('net_err').values/(.005*2.21857312*86400*5)
#xspec = np.loadtxt('emissionspec_corkT0.7002_ism.csv')[0]

# tab = read_file('merged_ltc_soft.fits')
# pha = tab.get_column('phase').values-0.5
# softrate = tab.get_column('counts').values/(.005*2.21857312*86400*5)
# softerate = np.sqrt(softrate)/((pha[-1]-pha[0])*.005*2.21857312*86400*5)
# #xspec = np.loadtxt('emissionspec_corkT0.7002_ism.csv')[0]
# print(rate)
# tab = read_file('merged_ltc_hard.fits')
# pha = tab.get_column('phase').values-0.5
# hardrate = tab.get_column('counts').values/(.005*2.21857312*86400*5)
# harderate = np.sqrt(hardrate)/((pha[-1]-pha[0])*.005*2.21857312*86400*5)
# #xspec = np.loadtxt('emissionspec_corkT0.7002_ism.csv')[0]
# print(harderate, hardrate)


def poissoncalc(simlc, rate=rate, pha=pha):
    simlc[1] = simlc[1]/simlc[1][0]*rate[0]
    inphalo = (simlc[0] > pha[0])
    inphahi = (simlc[0] < pha[-1])
    inpha = np.minimum(inphalo, inphahi).nonzero()
    newmin = inpha[0][0] - 1
    newmax = inpha[0][-1] + 1
    inpha = np.append(np.insert(inpha, 0, newmin), newmax)
    print(np.shape(simlc[1]))
    modpha = simlc[0][inpha]
    modlc = simlc[1][inpha]
    f = inter.interp1d(modpha, modlc)

    intermodlc = f(pha)

    #print(intermodlc)
    #print(rate)
    Plike = []

    for i, intermod in enumerate(intermodlc):
        tt = (round(intermod) ** round(rate[i])) / np.math.factorial(round(rate[i]))
        Plike.append(tt * np.exp(-round(intermod)))

    #print('a', intermodlc**rate)
    #print(Plike)
    print(np.prod(Plike))
    return intermodlc

def dothe(bspec, ax, rate=rate):


    erate = np.sqrt(rate)/((pha[-1]-pha[0])*.005*2.21857312*86400*5)
    oot = np.maximum((pha < 0.485), (pha > 0.515))
    ootcts = np.mean(rate[oot])

    tgrid = np.linspace(0.455, 0.545, 201)
    fullbest = np.sum(bspec, axis=-1)

    for index, aia in enumerate(fullbest):
        if index == 0:
            tot = np.zeros_like(aia)
            ax.plot(tgrid, aia, c='blue', alpha=0.25, label='Single AIA file')
        ax.plot(tgrid, aia, c='blue', alpha=0.25)
        # print(index)
        tot += aia
    tot /= 48
    try:
        inter = poissoncalc([tgrid, fullbest[0]])
    except:
        print('')

    ax.plot(tgrid, tot, c='gold', label='Average over \n60 AIA files')

    #ax.step(pha, inter, c='red', where='mid', marker='.', mfc='black', label='Stacked \nChandra data', mec='black')
    #ax.step(pha, rate/ootcts, c='red', where='mid', marker='.', mfc='black', label='Stacked \nChandra data', mec='black')
    #ax.errorbar(pha, rate/ootcts, erate/ootcts, c='black', ls='')
    ax.plot([0.485, 0.485], [np.min(fullbest), np.max(fullbest)], c='grey', ls='--', alpha=0.5)
    ax.plot([0.515, 0.515], [np.min(fullbest), np.max(fullbest)], c='grey', ls='--', alpha=0.5)
    #ax.legend(loc='lower left')
    #ax.set_ylabel('Normalised Count Rate')

    return ax

with np.load('eclipse_panes/data/sclht0.0502/dense89.0/abund11_naia48_ltcdata_splot.npz') as best:
    bspec = best['spec.npy'][:48]

# with np.load('eclipse_panes/data/sclht0.01/dense89.0/abund11_naia48_ltcdata_splot.npz') as lo:
#     lospec = lo['spec.npy'][:48]


# with np.load('eclipse_panes/data/sclht0.0702/dense1.0/abund11_naia48_ltcdata_splot.npz') as pop:
#     popspec = pop['spec.npy'][:48]

xspec = np.linspace(0.24, 7.007, len(bspec[0][0]))
soft_mask = np.minimum((xspec > 0.5), (xspec < 0.9))
softbest = popspec[:,:,soft_mask]*1
hard_mask = np.minimum((xspec > 0.9), (xspec < 7))
hardbest = popspec[:,:,hard_mask]*1

fig, ax = plt.subplots(1)
#ax[1] = dothe(bspec,ax[1])
#ax[0] = dothe(lospec,ax[0])
ax = dothe(hardbest,ax, rate)
#ax = dothe(hardbest,ax, hardrate)
#ax[2] = dothe(popspec,ax[2])

ax.set_xlabel('Orbital phase (0.5 is midtransit)')
ax.legend()

#ax[].set_xlim((0.455, 0.545))
#ax.loglog()
fig.text(0.04, 0.5, 'Flux', va='center', rotation='vertical')
#fig.tight_layout()
fig.savefig('plots/pop_hard_flux.pdf')
fig.show()
