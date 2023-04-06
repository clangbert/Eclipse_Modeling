from sherpa.astro.ui import *
import tqdm
import scipy.interpolate as inter
import scipy.special as sp
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from pycrates.hlui import *


def simp_chi_calc(simlc, nam, verbose=0):
    tab = read_file('merged_ltc_gehrels.fits')
    pha = tab.get_column('phase').values
    erate = tab.get_column('stat_err').values/(.005*2.21857312*86400)
    rate = tab.get_column('counts').values/(.005*2.21857312*86400)
    # Constrain simlc
    simlc[1] = simlc[1]/simlc[1, 0] * rate[0]
    if simlc[0][-1] < 1:
        simlc[0] += 0.5
    inphalo = (simlc[0] > pha[0])
    inphahi = (simlc[0] < pha[-1])
    inpha = np.minimum(inphalo, inphahi).nonzero()

    newmin = inpha[0][0] - 1
    newmax = inpha[0][-1] + 1
    inpha = np.append(np.insert(inpha, 0, newmin), newmax)
    modpha = simlc[0][inpha]
    modlc = simlc[1][inpha]

    f = inter.interp1d(modpha, modlc)

    intermodlc = f(pha)
    chi = np.sum(((intermodlc-rate)/erate)**2)

    if verbose > 0:
        plt.plot(simlc[0], simlc[1], label='Simulated data, chi={}'.format(round(chi,4)))
        plt.errorbar(pha, rate, erate, color="red", mfc="black", mec="black", ecolor="black", marker='.', ls='', label='X-ray Data')
        plt.title(nam)
        plt.legend()
        plt.xlabel('orbital phase')
        plt.ylabel('Count rate')
        plt.savefig('{}.png'.format(nam))
        plt.show()

    return chi


def poissoncalc(simlc, nam=None):

    def poiss_prob(actual, sim):
        # poiss = (round(intermod) ** round(rate[i])) / np.math.factorial(round(rate[i]))
        prob = np.exp(-sim)
        #iterative to control component sizes
        for i in range(actual):
            prob *= sim
            prob /= i+1

        return prob

    tab = read_file('merged_ltc_gehrels.fits')
    pha = tab.get_column('phase').values[:-2]
    erate = tab.get_column('stat_err').values[:-2]
    rate = tab.get_column('counts').values[:-2]

    out_pha = np.maximum((pha < 0.985), (pha > 1.015))

    # Constrain simlc
    simlc[1] = simlc[1]/simlc[1, 0] * np.mean(rate[out_pha])  # fudge factor on ltc to match counts from data
    if simlc[0][-1] < 1:
        simlc[0] += 0.5
    inphalo = (simlc[0] > pha[0])
    inphahi = (simlc[0] < pha[-1])
    inpha = np.minimum(inphalo, inphahi).nonzero()

    newmin = inpha[0][0] - 1
    newmax = inpha[0][-1] + 1
    inpha = np.append(np.insert(inpha, 0, newmin), newmax)
    modpha = simlc[0][inpha]
    modlc = simlc[1][inpha]

    f = inter.interp1d(modpha, modlc)

    intermodlc = f(pha)

    p_prob_list = []

    for index, value in enumerate(rate):
        p_prob_list.append(poiss_prob(value, intermodlc[index]))

    p_prob = np.prod(p_prob_list)

    if nam is not None:
        plt.plot(simlc[0], simlc[1], label='Simulated data, prob={}'.format(round(p_prob,4)))
        plt.errorbar(pha, rate, erate, color="red", mfc="black", mec="black", ecolor="black", marker='.', ls='', label='X-ray Data')
        plt.title(nam)
        plt.legend()
        plt.xlabel('orbital phase')
        plt.ylabel('Count rate')
        plt.savefig('plots/{}.png'.format(nam))
        plt.show()

    #print('a', intermodlc**rate)
    #print(Plike)
    #print(p_prob)

    log_p = np.sum([-intermodlc + rate*np.log(intermodlc) - sp.loggamma(rate+1)])

    return log_p
