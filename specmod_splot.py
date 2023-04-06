import numpy as np
import matplotlib.pyplot as plt
from sherpa.astro.ui import *
from scipy.integrate import trapz
import tqdm


msun = 1.989e+33  # Solar mass (grams)
rsun = 6.969e+10  # Solar radius (cm)
mjup = 1.8986e+30  # Jovian mass (grams)
rjup = 6.9911e+9  # Jovian radius (cm)
grav = 6.672e-8  # Gravitational constant (cm^3/gm/s^2)
kB = 1.380662e-16  # Boltzmann constant (erg/K)
degev = 8.6173468e-5  # Boltzmann constant (eV/K)
day = 86400  # Length of a day in seconds
kevang = 12.39852066  # keV * Ang (1e8*h*c/(e*1e10))
per = 2.21857312*day
amu = 1.166054e-24  # grams


def get_spec(path, e0, e1, component=1, corkT=None, coldepth=6*10**19):
    '''
    For getting the spectrum of unabsorbed light
    '''
    try:
        #fffff = 'g' + 6
        spec = np.loadtxt(path)
        print('loading spec from', path)
    except:
        # Simulate a spectrum
        clean()
        load_data('all_out_spec_src.pi')
        set_analysis(1, 'energy', 'counts')
        set_xsabund('grsa')

        cts, spec = get_sim_spec(coldepth=coldepth, erange=(e0[0], e1[0]), component=component, verbose=1, corkT=corkT)
        print(path)
        np.savetxt(path, spec)

    # Calculate total number of counts
    totcts = np.sum(spec[1])
    #plt.plot(spec[0], spec[1], label='first time{}'.format(component))

    # Calculate counts in each energy band
    bandcts = [0.] * len(e0)
    for index, emin in enumerate(e0):
        # Create a mask
        elt = (emin < spec[0])  # Low end
        egt = (e1[index] > spec[0])  # Higher end
        erange = np.minimum(elt, egt)  # Find the overlap values. Will be true for emin < spec < emax
        bandcts[index] = sum(spec[1][erange.nonzero()])

    #plt.plot(spec[0], spec[1], label='spec from mod {}'.format(component))
    return spec, totcts, bandcts


def get_sim_spec(coldepth=0, expo=116450.97187241, erange=(0.245, 7), verbose=0, component=1, corkT=None, abunds=[1,1,1]):
    """
    Simulates a spectral model based on the fit parameters of all_out & the elemental abundances in the atmosphere of
    the planet
    :param pabunds:
    :param verbose:
    :return:
    """
    comp = int(component)
    # clean()
    # set_xschatter(0)
    #load_data('all_out_spec_src.pi')
    create_model_component('xsvapec', 'mdl')
    # create_model_component('xsvapec', 'mdl2')
    ## Set the model
    # notice(erange[0], erange[1])

    set_source(1, (xsvphabs.abs2) * xsvapec.mdl)  # Source
    abs2.nH = coldepth / (10 ** 22)
    abs2.C, abs2.N, abs2.O = abunds
    #print(coldepth)

    with open('mg_tiedparams_allout_fitstats_xsvapec_cstat_v10_cts.txt') as f:
        lines = f.readlines()
    parnames = lines[5]
    parvals = lines[6]
    parnames = parnames.split('(')[1].split(')')[0].split(',')
    parvals = parvals.split('(')[1].split(')')[0].split(',')

    for index, nam in enumerate(parnames):
        nam = nam.split("'")[1]
        if nam == "mdl1.O":
            stellarO = parvals[index]
        elif nam == 'mdl1.Ne':
            stellarNe = parvals[index]
        elif nam == 'mdl1.Fe':
            stellarFe = parvals[index]
        elif nam == 'mdl1.Mg':
            stellarMg = parvals[index]
        elif nam == 'mdl{}.kT'.format(comp) and corkT is None:
            mdl.kT = parvals[index]
        elif nam == 'mdl{}.norm'.format(comp):
            mdl.norm = parvals[index]

    if corkT is not None:
        mdl.kT = corkT
    mdl.C, mdl.N, mdl.O = [stellarO] * 3
    mdl.Ne = stellarNe
    mdl.S, mdl.Si, mdl.Al, mdl.Mg = [stellarMg] * 4
    mdl.Ni, mdl.Ca, mdl.Ar, mdl.Fe = [stellarFe] * 4
    #show_model()

    notice(erange[0], erange[1])
    if verbose > 0:
        set_xlog()
        set_ylog()
        plot_model()
        plot_data(overplot=1)
        plot_source(overplot=True)
        plt.show()

    # sim_data = get_data_plot()
    # sim_xdat = sim_data.x
    # sim_ydat = sim_data.y

    sim_model = get_model_plot()
    sim_xmod = sim_model.x
    sim_ymod = sim_model.y

    splot= get_source_plot()
    xsplot = splot.x
    ysplot = splot.y
    inphalo = (xsplot >= 0.24)
    inphahi = (xsplot <= 7.007)
    mask = np.minimum(inphalo, inphahi)

    counts = sum(ysplot)
    # print(counts, 'vs', calc_model_sum(), 'vs', calc_source_sum(), 'vs', trapz(sim_ymod, sim_xmod))
    return counts, np.array([xsplot[mask], ysplot[mask]])


def get_atm_abs(star, plane, img2, nband, e0, e1, speclen, delt, frnt2, root, starspec):
    # Set up elemental abundances as per from getabund.pro with the values for Grevesse et al (1992).
    nx, ny = len(img2), len(img2[0])
    clean()
    specimgcts = np.array([[np.zeros_like(img2)] * 2] * nband)
    specimg = np.zeros((2, nx, ny, 676))
    xaxis = starspec[0][0]
    # print(np.shape(specimgcts))
    # print('delt', delt)
    # for band in range(nband):
    # elo, ehi = e0[band], e1[band]
    for comp, corkT in enumerate([star.corkT1, star.corkT2]):
        specabsorbfil = f'../../exoXtransit/absorb_spec_img/sclht{round(plane.sclht*rsun/rjup, 1)}/dense{plane.nbase / (1e+10)}/abund{plane.abundO}/notxs_{root}_specabsorb_{nx}x{ny}_corkT{corkT}'
        print(specabsorbfil)

        clean()
        set_xschatter(0)
        load_data('all_out_spec_src.pi')
        set_analysis(1, 'energy', 'counts')
        set_xsabund('grsa')
        expo_rat = 1/get_exposure()
        # Try to load an already calculated spec absorption image
        try:
            #frrrs = 'g'+6
            with np.load(specabsorbfil + '.npz') as data:
                specimg[comp] = data['arr_0.npy']
                print('loaded spec img')
            for band in range(nband):
                elo, ehi = e0[band], e1[band]
                specctfil = specabsorbfil + f'elo{elo}_ehi{ehi}.csv'
                specimgcts[band, comp] = np.loadtxt(specctfil)
                # print('Loaded spec absorb image from {}'.format(specabsorbfil))
            print('Loaded spec asorb')
        except:
            counts, modspec = get_sim_spec(img2[0][0], delt[0], [e0[0], e1[0]], component=comp + 1,
                                                       corkT=corkT)
            print(np.sum(modspec[1]))
            modspec[1] *= expo_rat
            print('expo test', expo_rat, np.sum(modspec[1]))
            specimg[comp] += modspec[1]
            specimg[comp][frnt2.nonzero()] *= 0

            #print('pixspec')
            counts = np.sum(modspec[1])
            specimgcts[0, comp][frnt2.nonzero()] *= 0
            specimgcts[0, comp] += counts
            for ban in range(nband - 1):
                band = ban + 1
                elo, ehi = e0[band], e1[band]
                #print('bands')
                elt = (elo < xaxis)  # Low end
                egt = (ehi > xaxis)  # Higher end
                #print('band')
                erange = np.minimum(elt,
                                    egt)  # Find the overlap values. Will be true for emin < spec < emax
                count = sum(modspec[1][erange.nonzero()])
                specimgcts[band, comp] += count
                specimgcts[band, comp][frnt2.nonzero()] *= 0
                #print(count)

            kx, ky = int(nx / 2), int(ny / 2)
            print('Calculating spec asorb')
            for x, row in tqdm.tqdm(enumerate(img2[:kx])):
                for y, coldepth in enumerate(row[:ky]):
                    if coldepth > star.dist and frnt2[x, y] == 0:
                        counts, modspec = get_sim_spec(coldepth, delt[0], [e0[0], e1[0]], component=comp + 1,
                                                       corkT=corkT, abunds=[plane.abundC, plane.abundN, plane.abundO])

                        # Radially symmetric so replicate per quarter
                        modspec[1] *= expo_rat
                        specimgcts[0, comp, x, y] = counts
                        specimgcts[0, comp, nx - x - 1, y] = counts
                        specimgcts[0, comp, x, ny - y - 1] = counts
                        specimgcts[0, comp, nx - x - 1, ny - y - 1] = counts
                        for ban in range(nband - 1):
                            band = ban + 1
                            elo, ehi = e0[band], e1[band]
                            elt = (elo < modspec[0])  # Low end
                            egt = (ehi > modspec[0])  # Higher end
                            erange = np.minimum(elt,
                                                egt)  # Find the overlap values. Will be true for emin < spec < emax
                            count = sum(modspec[1][erange.nonzero()])
                            specimgcts[band, comp, x, y] = count
                            specimgcts[band, comp, nx - x - 1, y] = count
                            specimgcts[band, comp, x, ny - y - 1] = count
                            specimgcts[band, comp, nx - x - 1, ny - y - 1] = count

                        specimg[comp, x, y] = modspec[1]
                        specimg[comp, nx - x - 1, y] = modspec[1]
                        specimg[comp, x, ny - y - 1] = modspec[1]
                        specimg[comp, nx - x - 1, ny - y - 1] = modspec[1]

                    # if x == y == 255:
                    # print('here')
                    ##plt.plot(modspec[0], modspec[1], label='comp{}'.format(comp+1))

                    ##plt.show()
                    ##print(sum(modspec[1]), specimgcts[x,y], specimgcts[x,y]/sum(modspec[1]))
                    ##return
            #print(np.shape(specimg))
            print('Saving absorption spec image as {}'.format(specabsorbfil))
            np.savez_compressed(specabsorbfil, specimg[comp])
            for band in range(nband):
                elo, ehi = e0[band], e1[band]
                specctfil = specabsorbfil + f'elo{elo}_ehi{ehi}.csv'
                np.savetxt(specctfil, specimgcts[band, comp])

    img2new = specimgcts * 1
    img2spec = specimg * 1

    return img2new, img2spec
