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
from pycrates.hlui import *

#from get_sherpa import get_sherpa

# NEEDS TO BE RUN IN SHERPA TERMINAL - OTHERWISE WON'T WORK
print('Successfully started')
# Define some constants (taken from inicon.pro)
msun = 1.989e+33  # Solar mass (grams)
rsun = 6.969e+10  # Solar radius (cm)
mjup = 1.8986e+30  # Jovian mass (grams)
rjup = 6.9911e+9  # Jovian radius (cm)
grav = 6.672e-8  # Gravitational constant (cm^3/gm/s^2)
kB = 1.380662e-16  # Boltzmann constant (erg/K)
degev = 8.6173468e-5  # Boltzmann constant (eV/K)
day = 86400  # Length of a day in seconds
kevang = 12.39852066  # keV * Ang (1e8*h*c/(e*1e10))
# per = 2.21857312*day
amu = 1.166054e-24  # grams
avg_dense = 1  # atoms cm^-3


# alternate amu def (i.e. atomic mass unit in grams)
#

def calcs(star, plane, vis=False, conecc=50, nx=512, ny=512, tbin=1000, rfunct='exp', useaia=False, nobs=20, npt=120, twoComponent=0, anim=False, naia=1, camera_val=0, lite=False):
    """
    Do the calculations
    :param star: The star objectprint(shape(img2))
    plt.imshow(img2, norm=LogNorm())
    plt.colorbar()
    #plt.savefig('absoec test.png')
    plt.show()
    return
    :param plane: The planet object
    :param vis: Display or not, default is False
    :param conecc: Cone angle (default is 50 degrees)
    :param nx: Number of x bins (default is 512)
    :param ny: Number of y bins (default is 512)
        plot_data(overplot = 1)
    :param tbin: Time bins for the light curve (default is 1000)
    :param rfunct: The radially exponential density drop, defaults to 'exponential'
    :param useaia: Tells the function whether or not to load in AIA images
    :param nobs: Number of observations (defaults to 20)
    :param npt: Number if points with which to cover each eclipse (defaults to 120)
    """
    naia = naia if useaia else 1
    # Energy bands
    if lite:
        pbands = pd.read_csv('passbands.csv') # added by cgl for running lite code
    else:
        pbands = pd.read_csv('passbands2.csv') 
    e0 = pbands['e0'].values  # [0.5, 0.5, 0.8, 1.2, 2.3]#, 0.5, 0.3, 0.5, 0.2]
    e1 = pbands['e1'].values  # [7.0, 0.8, 1.2, 2.3, 7]#, 1.5, 1.7, 1.7, 2.3]
    lbands = pbands['lbands'].values  # ['full', 'soft', 'medium', 'hard', 'harder']#, 'r315', 'r515', 'r317', 'r517', 'SEEJ']
    cbands = pbands['cbands'].values  # ['0.5-7.0', '0.5-0.8', '0.8-1.2', '1.2-2.3', '2.3-7']#, '0.5-1.5', '0.3-1.7', '0.5-1.7', '0.2-2.3']
    emin, emax = np.min(e0), np.max(e1)
    nband = len(e0)
    cwrange = str(round(kevang / emax, 2)) + '-' + str(round(kevang / emin, 2))

    # Define the file name
    root = "{}_corkT1{}_corkT2{}_abundO{}_corht1{}_corht2{}_CNO{}_Fe{}_base{}_cone{}_nx{}_ny{}_rfunct{}_sclht{}_naia{}".format(star.name, round(star.corkT1, 4),
                                                                              round(star.corkT2, 4), round(plane.abundO, 4),
                                                                              round(star.coronaht1, 4), round(star.coronaht2, 4),
                                                                              round(plane.abundC, 4),
                                                                              round(plane.abundFe, 4),
                                                                              round((plane.nbase/1e+10), 4),
                                                                              round(conecc, 4), nx, ny, rfunct, round(plane.sclht, 4), naia)

    # Extract masses, radii and heights
    m1, m2 = star.mass, plane.mass
    r1, r2 = star.rad, plane.rad
    h1, h2, h3 = star.coronaht1, plane.sclht, star.coronaht2
    print(h1, h2)
    inclin = plane.inclin * 180 / np.pi
    # Kepler's 3rd for semi diamater of orbit (Budding et al 1996, Astrophys. Space Sci., 236, 215)
    semiA = ((star.per ** 2) * (m1 + m2) / 0.0134) ** 0.33333
    # radii of orbits for each of the components A1 and A2
    semiA1 = m2 * 2 * semiA / m1 / (1 + m2 / m1)
    semiA2 = m1 * 2 * semiA / m2 / (1 + m1 / m2)

    # Set the ranges in real units for x and y
    xra = [-(2. * semiA + r1 + r2) * 1.2 / 2., (2. * semiA + r1 + r2) * 1.2 / 2.]
    yra = xra
    print('xra', xra)
    # Set up grids
    #phgrid = 0.15 * np.linspace(0, npt, npt + 1) / npt
    tgrid = np.linspace(0.455, 0.545, npt)
    #thi = (tgrid + 0.5 < 1.045)
    #tlo = (tgrid + 0.5 > 0.955)
    #tok = np.minimum(tlo, thi)
    #tgrid = tgrid[tok]
    times = tgrid * star.per * day
    # Work out tbounds -- dont have mid2bound and since halfit called will start by calculating the midpoints between
    # each point as the average of the points either side
    tbounds = (times[1:] + times[:-1]) / 2
    # 2 elements too short -- missing begin and end, incorporate by assuming that first and last bin extend just as far
    # in either direction i.e. difference between 0th and 1st is half the total width of 0th
    tbounds = np.append(np.array([tbounds[0] - (tbounds[1] - tbounds[0])]), tbounds)
    tbounds = np.append(tbounds, np.array([tbounds[-1] + (tbounds[-1] - tbounds[-2])]))
    delt = tbounds[1:] - tbounds[:-1]
    nt = len(tgrid)

    # See if already calculated
    # if lite:
    datapdname = f'../../exoXtransit/eclipse_panes/data/sclht{round(plane.sclhtRJ, 5)}/dense{plane.nbase/(1e+10)}/abund{plane.abundO}_srad{star.rad}_mrad{star.mass}_naia{naia}_npt{npt}_inclin{inclin}_corkT1{round(star.corkT1, 4)}_corkT2{round(star.corkT2, 4)}.fits'
    # else:
    #     datapdname = f'../../exoXtransit/eclipse_panes/data/sclht{round(plane.sclhtRJ, 5)}/dense{plane.nbase/(1e+10)}/abund{plane.abundO}_naia{naia}_corkT1{round(star.corkT1, 4)}_corkT2{round(star.corkT2, 4)}.fits'
    
    try:

        print('attempting to load files', datapdname)
        with fits.open(datapdname) as data:
            print('file found')
            lcave_cts = data['BROAD_CTS'].data['AVG']
            lcave_src = data['BROAD_SRC'].data
            lcfull = data['broad_cts'].data
            lcsummed = data['MED_CTS'].data
            print(f'Files loaded successfully from {datapdname}')
        return lcave_cts, lcfull, lcave_src, lcsummed, tgrid, False
    except:
        print('ltc file not found for these params, evaluating.')

    # Generate emission spectrum from star with no absorption
    specpath1 = 'emissionspec_corkT{}_'.format(round(star.corkT1, 4))
    spec1, totcts1, bandcts1, src1 = get_spec(specpath1, e0, e1, component=1, corkT=star.corkT1, coldepth=star.dist)
    specpath2 = 'emissionspec_corkT{}_'.format(round(star.corkT2, 4))
    spec2, totcts2, bandcts2, src2 = get_spec(specpath2, e0, e1, component=2, corkT=star.corkT2, coldepth=star.dist)
    print(totcts1, totcts2)
    specarr = [spec1[1], spec2[1]]
    specx = spec1[0]*1
    srcx = src1[0]*1
    totctsarr = [totcts1, totcts2]
    print(len(specx), 'specx len')


    ### Generate images for each component of the star  -- NOW DONE WHEN DEFINING THE STAR OBJECTS
    #solimg1 = get_star_img(r1, h1, vis, xra, yra, nx, ny, rfunct, conecc, star, star.corkT1, useaia=useaia)
    #solimg2 = get_star_img(r1, h3, vis, xra, yra, nx, ny, rfunct, conecc, star, star.corkT2, useaia=useaia)

    #star.solimgarr = np.array([star.solimg1, star.solimg2])

    # Planet image
    print('getting atmosphere', plane.sclhtRJ)
    img2 = sphrojcol(r2, ourad=r2+10*h2, verbose=10, xrange=xra, yrange=yra, nx=nx, ny=ny, sclht=h2, rfunct=rfunct, coneang=conecc,
                     root=plane.name+root, nbase=plane.nbase, libdir=f'../../exoXtransit/sphrojcolfils/sclht{round(plane.sclhtRJ, 5)}/dense{plane.nbase/1e+10}/bigggerrr')  # old ourad = r2+8.5*h2
    print(img2.max()/rjup)

    img2 *= plane.nbase
    #return img2, plane.nbase
    print('including ism_col')
    ism_col = star.dist * avg_dense
    print(ism_col, np.max(img2))
    ism_col_img = np.zeros_like(img2) + ism_col

    img2 += ism_col  # np.maximum(img2, ism_col_img)

    print('Compare column down to middle of planet vs expected:\n', img2[int(nx/2), int(ny/2)],
          plane.sclht*plane.nbase*rsun)  # Vastly different than expected... (well, an order of 2 different)  # Nope i was just not converting units properly

    print('out of sphrojcol')

    print('atmosphere ratio', img2[img2.nonzero()].min()/img2.max()*100)
    centralcol = img2[int(nx/2), int(ny/2)]

    # Set up the circle fronts for star and planet
    x0, y0 = 0, 0  # Origin coords
    frnt1 = ccircle(x0, y0, r1, xrange=xra, yrange=yra, nx=nx, ny=ny)  # Star
    frnt2 = ccircle(x0, y0, r2, xrange=xra, yrange=yra, nx=nx, ny=ny)  # Planet

    core = np.transpose(frnt2.nonzero())
    atmsize = len(np.transpose(img2.nonzero())) - len(core)
    speclen = len(specx)
    absspecarr = np.zeros([2, atmsize, len(spec1[0])])
    summedspec = np.zeros([2, len(spec1[0])])
    print('getting atm abs')
    img2, img2spec_cts, img2spec_src = get_atm_abs(star, plane, img2, nband, e0, e1, speclen, len(srcx), delt, frnt2, 'bigerrrr', starspec=[spec1, spec2])
    #print('some sums', np.max(np.sum(img2spec, axis=-1)), np.sum(img2), np.shape(img2spec))

    # generate fits file
    if lite:
        fits_gen_lite(specx, srcx, naia, tgrid, star.solimgarr, img2spec_src, img2spec_cts, datapdname)
    else:
        fits_gen(specx, srcx, naia, tgrid, star.solimgarr, img2spec_src, img2spec_cts, datapdname)


    #plt.imshow(np.sum(img2spec[1], axis=-1))
    #plt.savefig('25v2.png')
    #plt.show()

    #camera = Camera(fig)
    # Iterate over time
    clean()
    tforlc = time.time()
    #specarr = np.array([img2spec[0][0][0], img2spec[1, 0, 0]])
    print(np.shape(specarr))
    lcsummed_cts, lcave_cts, lc_count, lcsummed_src, lcave_src, lc_src = transit(tgrid, naia, star, plane, img2, specx, srcx, img2spec_src, img2spec_cts, nx=nx, ny=ny, anim=anim, e0=e0, e1=e1, root=datapdname, verbose=camera_val, lite=lite)
    print('t to calc lc', time.time() - tforlc)
    lcfull = lcsummed_cts[0]*1



    #np.savez_compressed(datapdname, bands=lcsummed, cts=lcfull, avg=lcave, comp=components)

    return lcave_cts, lcfull, lcsummed_src, lcsummed_cts, tgrid, True
