import gc
from matplotlib.colors import LogNorm
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import math as ma
import tqdm
from sphrojcol import sphrojcol
from sphroj import sphroj
from ccircle import ccircle
import mendeleev
import scipy.interpolate as inter
import pandas as pd
import glob
import _pickle as pickle
import glymur
from congrid import congrid
from AIAFile import AIAFile
from celluloid import Camera
import time
import sys
from sherpa.astro.ui import *
from glob import glob
from astropy.io import fits


def calc_pix_counts(starimg, specimg, pabund, tstep, modspec, erange=(0.5, 7)):
    #print('here')
    #tstart = time.time()
    absorbimg = starimg*1
    #totstar = np.sum(starimg)
    noabscounts = np.sum(modspec[1][np.minimum((erange[0] < modspec[0]), (erange[1] > modspec[0])).nonzero()])
    absorbimg *= noabscounts
    #print(np.sum(absorbimg), noabscounts)
    #print('time to evaluate noabs', time.time() - tstart)
    #tstart = time.time()
    """for x, row in enumerate(starimg*1):
        for y, intense in enumerate(row):
            specdepth = specimg[x, y]*1
            if specdepth != 0:
                absorbimg[x, y] = specdepth * intense
    """          #print(absorbimg[x, y])
    whereabs = specimg.nonzero()
    absorb = (1*starimg)[whereabs] * (specimg*1)[whereabs]
    absorbimg[whereabs] = absorb
    #print('time to evaluate abs', time.time() - tstart)
    #absorbimg *= tstep
    return absorbimg


def calc_pix_spec(star_spec_img, star_spec_img_copy, spec, comp, planet_start, planet_now, strip, core):
    # Multiply spectrum by star image
    star_spec_img_copy[strip[0]-1:strip[1]+1] = (star_spec_img[strip[0]-1:strip[1]+1])*spec[comp, 0, 0]
    star_spec_img_copy[planet_now] = star_spec_img[planet_now] * spec[comp][planet_start]
    star_spec_img_copy[core.nonzero()] *= 0.
    #print('generating flux_img_step_spec', time.time() - ts, np.shape(star_spec_img_copy))
    #print(np.shape(star_spec_img_copy))
    inst_spec = np.sum(star_spec_img_copy, axis=(0,1))
    return star_spec_img_copy, inst_spec


def saving_spec(spec, path, tab_nam, time_step):
    new_col =  fits.ColDefs([fits.Column(name=str(time_step), format='D', array=spec)])

    hdul = fits.open(path)
    aia_tab = hdul[tab_nam].data
    orig_cols = aia_tab.columns

    new_tab = fits.BinTableHDU.from_columns(orig_cols + new_col, name=tab_nam)
    hdul[tab_nam] = new_tab
    hdul.writeto(path, overwrite=True)
    hdul.close()


def transit(tgrid, naia, star, plane, img2, specx, xsrc, src_spec, count_spec, nx=512, ny=512, anim=False, e0=[0.5], e1=[7], root='test', verbose=0, lite=False):
    nt = len(tgrid)
    #specx = np.linspace(0.24,7.007,676)
    if verbose > 0:
        fig = plt.figure(figsize=(18.8, 4.8))
        ax = fig.subplot_mosaic("""
        AABBCC
        AABBDD
        """)
        camera = Camera(fig)
        ax['C'].loglog()
        ax['A'].set_xlabel('Orbital phase (0.5 is mid transit)')
        ax['A'].set_ylabel('Normalised Counts per second')
        ax['B'].set_xticks([]); ax['B'].set_yticks([])
        ax['C'].set_xticklabels([])
        ax['C'].set_ylabel('Counts per second')
        ax['D'].set_ylabel('Instantaneous spectum /\nOut of transit')
        ax['D'].set_xlabel('Energy (keV)')

    # Extract masses, radii and heights
    m1, m2 = star.mass, plane.mass
    r1, r2 = star.rad, plane.rad
    h1, h2, h3 = star.coronaht1, plane.sclht, star.coronaht2
    print(h1, h2)
    # Kepler's 3rd for semi diamater of orbit (Budding et al 1996, Astrophys. Space Sci., 236, 215)
    semiA = ((star.per ** 2) * (m1 + m2) / 0.0134) ** 0.33333
    # radii of orbits for each of the components A1 and A2
    semiA1 = m2 * 2 * semiA / m1 / (1 + m2 / m1)
    semiA2 = m1 * 2 * semiA / m2 / (1 + m1 / m2)

    # Set up band masks
    masklist = [0]*len(e0)
    mask_src = [0]*len(e0)
    for bandind, band in enumerate(e0):
        masklist[bandind] = np.minimum((specx >= band), (specx < e1[bandind]))
        mask_src[bandind] = np.minimum((xsrc >= band), (xsrc < e1[bandind]))

    # Set the ranges in real units for x and y
    xra = [-(2. * semiA + r1 + r2) * 1.2 / 2., (2. * semiA + r1 + r2) * 1.2 / 2.]
    yra = xra
    print('xra', xra)

    # Set up the circle fronts for star and planet
    x0, y0 = 0, 0  # Origin coords
    frnt1 = ccircle(x0, y0, r1, xrange=xra, yrange=yra, nx=nx, ny=ny)  # Star
    frnt2 = ccircle(x0, y0, r2, xrange=xra, yrange=yra, nx=nx, ny=ny)  # Planet

    planimgs = np.sum(count_spec[0], axis = -1)  # useful for knowing extent of atm
    mincounts = planimgs[0][0]
    wherepla = (planimgs < mincounts).nonzero()
    where_planet_start = (planimgs < mincounts)

    whereplaOG = wherepla*1
    #planimgs[wherepla] = planimgs[wherepla]/planimgs[wherepla]

    nfil = len(star.solimgarr[0])
    # Set up light curves
    lc_count = np.zeros([2, naia, len(e0), nt])  # Output light curves including absorption, 1 per band
    lc_src = lc_count*1
    #  lcfull = np.zeros((2, nfil, nt))
    if anim is True:
        lcspec = np.zeros([2, nt, len(specx)])
    else:
        lcspec = np.zeros([2, 5, len(specx)])
    # lc0 = np.zeros([2, nfil, nt])  # Light curve only considering direct planet opacity and not extended atmosphere
    # lcgeom = np.zeros(nt)  # Light curve for flat image, useful for first and fourth contact points

    # Calculating now positions over time. Assume phase 0 is such that component 1 is directly in front and moving to
    # the right. Star in front has -ve y. Looping over time (phase)

    dx = (xra[1] - xra[0]) / (nx - 1.)
    dy = (xra[1] - yra[0]) / (ny - 1.)

    # Grid of positions
    x1grid = semiA1 * np.sin(2. * np.pi * tgrid)
    y1grid = np.cos(plane.inclin + np.pi) * semiA1 * np.cos(2 * np.pi * tgrid)
    ix1grid = (x1grid - x0) / dx + (nx / 2)
    iy1grid = (y1grid - y0) / dy + (ny / 2)

    y2grid = -semiA2 * np.sin(2 * np.pi * tgrid + np.pi)
    x2grid = -np.cos(plane.inclin) * semiA2 * np.cos(2 * np.pi * tgrid)
    ix2grid = (x2grid - x0) / dx + (nx / 2)
    iy2grid = (y2grid - y0) / dy + (ny / 2)

    platop = np.min(whereplaOG[0])
    plabot = np.max(whereplaOG[0])
    pla_rad = int((plabot-platop)/2)

    where_frnt = frnt1.nonzero()
    statop = np.min(where_frnt[0])
    stabot = np.max(where_frnt[1])

    bandbot = 512-(int((ix2grid.min())) - pla_rad)#))
    bandtop = 512-(int(ix2grid.max()) + pla_rad)#)
    print(bandtop, bandbot)
    #return
    fluximgarr = np.zeros([2, nt, nx, ny])  # Array for storing fluximages at every time step

    atmcheck = []
    print('entering loop')
    count_spec_sum = np.sum(count_spec, axis=-1)
    count_spec_cop = np.copy(count_spec)

    src_spec_sum = np.sum(src_spec, axis=-1)
    src_spec_cop = np.copy(src_spec)
    #ax.imshow(count_spec_sum[0])
    #print(np.shape(count_spec_sum), np.max(count_spec_sum))
    #fig.savefig('plots/quick check.png')
    cb=False

    hdul = fits.open(root)
    for k in tqdm.tqdm(range(naia)):  #range(naia):#
        for comp, solimg in enumerate(star.solimgarr):
            #fluximgarr += solimg[k]*count_spec_sum[comp][0][0]
            star_spec_img = solimg[k].reshape(nx, ny, 1)
            star_spec_img_count = star_spec_img*count_spec_cop[comp, 0, 0]
            star_spec_img_src = star_spec_img*src_spec_cop[comp, 0, 0]

            aia_tab_src = hdul[f'AIA{k}_comp{comp+1}_src'].data
            orig_cols_src = aia_tab_src.columns
            new_col_src = []

            aia_tab_count = hdul[f'AIA{k}_comp{comp+1}_count'].data
            orig_cols_count = aia_tab_count.columns
            new_col_cts = []

            for index, t in enumerate(tgrid):  #tqdm.tqdm(
                ts=time.time()
                ix2, iy2 = ix2grid[index], iy2grid[index]
                # Move the planet
                planet_core = np.roll(frnt2, (int(nx / 2 - ix2), int(ny / 2 - iy2)), (0, 1))
                where_planet = np.roll(where_planet_start, (int(nx / 2 - ix2), int(ny / 2 - iy2)), (0, 1))

                '''# Multiply spectrum by star image
                star_spec_img_copy[bandtop-1:bandbot+1] = (star_spec_img[bandtop-1:bandbot+1])*count_spec_cop[comp, 0, 0]
                star_spec_img_copy[where_planet] = star_spec_img[where_planet] * count_spec_cop[comp][where_planet_start]
                #print('generating flux_img_step_spec', time.time() - ts, np.shape(star_spec_img_copy))
                ts=time.time()
                #return

                # Obscure where the planet's core is
                star_spec_img_copy[planet_core.nonzero()] *= 0.'''
                # star_spec_img, star_spec_img_copy, spec, comp, planet_start, planet_now, strip, core
                star_spec_img_src, inst_spec_src = calc_pix_spec(star_spec_img, star_spec_img_src, src_spec_cop, comp,
                                                                 where_planet_start, where_planet, [bandtop, bandbot],
                                                                 planet_core)

                star_spec_img_count, inst_spec_count = calc_pix_spec(star_spec_img, star_spec_img_count, count_spec_cop, comp,
                                                                     where_planet_start, where_planet, [bandtop, bandbot],
                                                                     planet_core)

                #spec, path, tab_nam, time_step

                if not lite:
                    # save individual spectra
                    new_col_src.append(fits.Column(name=str(t), format='D', array=inst_spec_src))
                    new_col_cts.append(fits.Column(name=str(t), format='D', array=inst_spec_count))


                #print('img shape', np.shape(fluximgarr))
                if k == 0:
                    fluximgarr[comp, index] = np.sum(star_spec_img_count, axis=-1)

                for maskind, mask in enumerate(masklist):
                    lc_count[comp, k, maskind, index] = np.sum(inst_spec_count[mask])
                    lc_src[comp, k, maskind, index] = np.sum(inst_spec_src[mask_src[maskind]])

                # Make a movie for the first AIA image (k=0)
                if k == 0 and verbose > 0:

                    if anim is False and index % int(nt/2) == 0:
                        lcspec[comp, int(index/(nt/2))] = inst_spec_count*1
                    elif anim is True:
                        lcspec[comp, index] = inst_spec_count*1

                    # print('here')
                    if comp == 1:
                        if index == int(nt / 2) or (anim is True and index % 3 == 0):
                            #print('image')
                            cts_img = ax['B'].imshow(fluximgarr[0, index] + fluximgarr[1, index], norm=LogNorm(vmin=1e-15))#, norm=LogNorm(vmin=1e-8)) #['B']
                            if cb:
                                fig.colorbar(cts_img)
                                cb = False
                            #print('spec')
                            spechere = lcspec[0, int(index/(nt/2))]+lcspec[1, int(index/(nt/2))] if anim is False else lcspec[0, index]+lcspec[1, index]
                            ax['C'].plot(specx, spechere, c='black')
                            #print('instspec done')
                            ax['C'].loglog()
                            ax['D'].plot(specx, spechere/(lcspec[0, 0]+lcspec[1, 0]), c='red')
                            #print('rel spec done')
                            ax['D'].set_xscale('log')
                            if index == int(nt/2):
                                np.savetxt('midtransitpanel.csv', fluximgarr[0, index] + fluximgarr[1, index])
                        if (anim is True and index % 3 == 0) or (anim is False and index==nt-2):
                            #print('ltc')
                            #print((lc_count[0, k, 0, index] + lc_count[1,k,index])/(lcfull[0,k,0]+lcfull[1,k,0]))
                            #print(np.shape(lc))
                            ax['A'].plot(tgrid[:index],
                                    (lc_count[0, k, 0,:index] + lc_count[1, k, 0, :index])/((lc_count[0, k, 0, 0] + lc_count[1, k, 0, 0])),
                                    c='blue')
                            #print('lc done')
                            camera.snap()
                    #atmcheck.append((fluximgarr[0, index][255, 255] + fluximgarr[1, index][255, 255]))



                #print('plotting', time.time()-ts)
                ts=time.time()
                #return
            if not lite:
                # append individual transit light curves
                new_tab = fits.BinTableHDU.from_columns(orig_cols_src + fits.ColDefs(new_col_src), name=f'AIA{k}_comp{comp+1}_src')
                hdul[f'AIA{k}_comp{comp+1}_src'] = new_tab
                new_tab = fits.BinTableHDU.from_columns(orig_cols_count + fits.ColDefs(new_col_cts), name=f'AIA{k}_comp{comp+1}_count')
                hdul[f'AIA{k}_comp{comp+1}_count'] = new_tab

    lcsummed_cts = lc_count[0] + lc_count[1]
    lcsummed_src = lc_src[0] + lc_src[1]
    lcave_cts = np.sum(lcsummed_cts, axis=0)/naia
    lcave_src = np.sum(lcsummed_src, axis=0)/naia

    if verbose > 0:
        if anim is True:
            print('saving the animation')
            animation = camera.animate()
            animation.save(f'../../exoXtransit/plots/ht{plane.sclht}_den{plane.nbase/1e10}_abund{plane.abundO}_rstar{star.rad}_inc{plane.inclin*180/np.pi}_subplots.gif')
        else:
            #ax['A'].plot(tgrid, lcfull[0, 0]/lcfull[0,0,0], label='Norm after add')
            ## ax['A'].plot(tgrid, (lc_count[0, k, 0]/lc_count[0, 0, 0, 0]+lc_count[1, k, 0]/lc_count[1, 0, 0, 0])/2, label='norm before add')
            ##ax['A'].legend()
            #ax['A'].set_ylabel('Normalised Counts')
            #ax['A'].set_xlabel('Phase')
            fig.suptitle('Test Eclipse. OAbund={}, psclht={}, pdense={}'.format(round(plane.abundO,4), round(plane.sclht,4), round(plane.nbase/1e+11, 4)))
            fig.savefig(f'../../exoXtransit/plots/sclht{plane.sclht}_dense{plane.nbase/1e+10}_abund{plane.abundO}_midtransit.png')
        fig.show()
        plt.show()
        plt.close()

        plt.plot(atmcheck)
        plt.savefig('atmcheck.png')

    # Save to file
    if lite:
        bnames=['BROAD', 'MED']
    else:
        bnames=['BROAD', 'USOFT', 'SOFT', 'MED', 'HARD']
    
    # SRC_BANDS
    for ind, bname in enumerate(bnames):
        if lite:
            new_col = fits.Column(name=f'AVG', format='D', array=lcave_src[ind]) # added by cgl
        else:
            collist = [fits.Column(name=f'AIA{i}', format='D', array=lcsummed_src[i][ind]) for i in range(naia)]
            collist.insert(0, fits.Column(name=f'AVG', format='D', array=lcave_src[ind]))
            new_col = fits.ColDefs(collist)
        avg_src = hdul[f'{bname}_src'].data
        orig_cols = avg_src.columns
        new_tab = fits.BinTableHDU.from_columns(orig_cols + new_col, name=f'{bname}_src')
        hdul[f'{bname}_src'] = new_tab
    # CTS_BANDS
    for ind, bname in enumerate(bnames):
        if lite:
            new_col = fits.Column(name=f'AVG', format='D', array=lcave_cts[ind]) # added by cgl
        else:
            collist = [fits.Column(name=f'AIA{i}', format='D', array=lcsummed_cts[i][ind]) for i in range(naia)]
            collist.insert(0, fits.Column(name=f'AVG', format='D', array=lcave_cts[ind]))
            new_col =  fits.ColDefs(collist)
        avg_cts = hdul[f'{bname}_cts'].data
        orig_cols = avg_cts.columns
        new_tab = fits.BinTableHDU.from_columns(orig_cols + new_col, name=f'{bname}_cts')
        hdul[f'{bname}_cts'] = new_tab
    
    # else:
    #     # SRC_BANDS
    #     for ind, bname in enumerate(bnames):
    #         collist = [fits.Column(name=f'AIA{i}', format='D', array=lcsummed_src[i][ind]) for i in range(naia)]
    #         collist.insert(0, fits.Column(name=f'AVG', format='D', array=lcave_src[ind]))
    #         new_col = fits.ColDefs(collist)
    #         # new_col = fits.Column(name=f'AVG', format='D', array=lcave_src[ind]) # added by cgl
    #         avg_src = hdul[f'{bname}_src'].data
    #         orig_cols = avg_src.columns
    #         new_tab = fits.BinTableHDU.from_columns(orig_cols + new_col, name=f'{bname}_src')
    #         hdul[f'{bname}_src'] = new_tab
    #     # CTS_BANDS
    #     for ind, bname in enumerate(bnames):
    #         collist = [fits.Column(name=f'AIA{i}', format='D', array=lcsummed_cts[i][ind]) for i in range(naia)]
    #         collist.insert(0, fits.Column(name=f'AVG', format='D', array=lcave_cts[ind]))
    #         new_col =  fits.ColDefs(collist)
    #         # new_col = fits.Column(name=f'AVG', format='D', array=lcave_cts[ind]) # added by cgl
    #         avg_cts = hdul[f'{bname}_cts'].data
    #         orig_cols = avg_cts.columns
    #         new_tab = fits.BinTableHDU.from_columns(orig_cols + new_col, name=f'{bname}_cts')
    #         hdul[f'{bname}_cts'] = new_tab


    hdul.writeto(root, overwrite=True)
    hdul.close()
    return lcsummed_cts, lcave_cts, lc_count, lcsummed_src, lcave_src, lc_src






        ## AVG_SRC
        #new_col =  fits.ColDefs([[fits.Column(name=f't={tgrid[i]}', format='D', array=lcave_src[i])] for i in range(len(tgrid)]])
        #avg_src = hdul['AVG_SRC'].data
        #orig_cols = avg_src.columns
        #new_tab = fits.BinTableHDU.from_columns(orig_cols + new_col, name=tab_nam)
        #hdul['AVG_SRC'] = new_tab
        ## AVG_CTS
        #new_col =  fits.ColDefs([[fits.Column(name=f't={tgrid[i]}', format='D', array=lcave_cts[i])] for i in range(len(tgrid)]])
        #avg_src = hdul['AVG_CTS'].data
        #orig_cols = avg_src.columns
        #new_tab = fits.BinTableHDU.from_columns(orig_cols + new_col, name=tab_nam)
        #hdul['AVG_CTS'] = new_tab
        #bnames=['BROAD', 'USOFT', 'SOFT', 'MED', 'HARD']
