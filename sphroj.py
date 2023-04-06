import os
import random
import numpy as np
import matplotlib.pyplot as plt
import math as ma
import tqdm
from pathproj import pathproj
from ccircle import ccircle


def sphroj(inrad, ourad, drad=None, xrange=None, yrange=None, nx=512, ny=512, root=None, libdir='./',
              verbose=False, early=False, rfunct='exp', sclht=None, coneang=90, halfsh=None, norm=False, small=False,
              rand=False, nstep=1001):
    """
    Function:
        Sphrojcol:  Calculates 2D image projection of a possibly hollow spherical volume with radially varying number
                    density profile by calculating integrals of column density along line of sight

    Parameters:
        :param inrad:   Inner radius of the sphere
        :param ourad:   Outer radius of the sphere
        :key drad:      Thickness of each slice (default is (ourad-inrad)/20)
        :key xrange:    Range of x values (default is [-2*ourad, 2*ourad])
        :key yrange:    Range of y values (default is [-2*ourad, 2*ourad])
        :key nx:        Number of x elements (default is 512)
        :key ny:        Number of y elements (default is 512)
        :key root:      Look for previously saved files named ROOT_NNNNxNNNN_FFFF_FFFFF_RRRRR_SSSSS_sCCC.csv
                        where NNNNxNNNN == NX,NY, FFFF_FFFF is INRAD_OURAD in f5.3,
                        RRRRR is 'exp'/'const'/'invsq' is the type of radial
                        dependency, SSSS is the scale height in f5.3, and sCCC is cone angle and its sign (p/m).
                        ROOT should include HALFSH and NSTEP keywords.
                        Ignored by default.
        :key libdir:    Looks for library files in this directory. Ignored if root not set. Default is './'.
                        NOTE: if the appropriate library save file:
                            --  exists in LIBDIR: then it will be read in and
                                the call to PATHPROJ will be skipped.
                            --  does not exist in LIBDIR: the image calculated
                                with PATHPROJ will be written out to appropriately
                                named save file in LIBDIR
        :key verbose:   Boolean, False by default.
                        If set to true, *lots* of plots will be made
        :key early:        Boolean, False by default
                        If True, will return a blank image of 0s nx by ny in size.
        :key rfunct:    Function describing number density variation for pathproj. Default is 'exp'.
        :key sclht:     Scale-height for exponential drop-off.
                        Default is abs(ourad-inrad) if abs(ourad-inrad) > 1 else 1.
        :key coneang:   Opening cone angle. Default is +90 degrees.
        :key halfsh:    Catch and discard. pathproj will use full shell
        :key norm:      Normalisation factor for ccircle, default to False.
        :key small:     Tells ccircle whether or not to do small plot. Default to False.
        :key rand:      Tells ccircle whether or not to randomise intensity. Default to False.
        :key nstep:     Tells pathproj how many steps to use in its integral. Default to 1001.
    :return: 2D image projection of (possibly hollow) spherical volume of varying density

    Dependencies:
        pathproj
        ccircle
        tqdm  # replacing kilroy

    History:
        [28/09/2021]    File created by Joseph Hall adapting from V. Kashyap's sphroj.pro. Based on sphrojcol.py.
    """
    # Set constants:
    RSun = 6.957e+10  # Radius of the sun in cm

    # Check inputs
    if ourad <= inrad:
        print('outer radius smaller than inner?')
        return 0*ccircle(0, 0, 1)

    # Set up keywords, starting with ranges
    delr = drad if drad is not None else (ourad-inrad)/20
    xr = xrange if xrange is not None else [-2 * ourad, 2 * ourad]
    yr = yrange if yrange is not None else [-2 * ourad, 2 * ourad]
    # Make sure ranges go from low to high not vice versa and that the elements are not identical
    xr, yr = np.sort(xr), np.sort(yr)
    if xr[0] == xr[1]:
        xr[1] += 1
    if yr[0] == yr[1]:
        yr[1] += 1
    # Set up bin sizes
    binx = abs(int(nx)) if abs(nx) > 1 else 1
    biny = abs(int(ny)) if abs(ny) > 1 else 1
    # Pixel size(?)
    xpix, ypix = (xr[1] - xr[0])/binx, (yr[1] - yr[0])/biny

    # Use pathproj keywords to define savefile filename
    rrrr = rfunct
    sss = sclht if sclht is not None else ((abs(ourad-inrad)) if abs(ourad-inrad) > 1 else 1)
    ssss = int(sss*1000)
    cc = coneang if coneang is not None else 90
    ss = 'p' if coneang >= 0 else 'm'
    ccc = ss + str(abs(int(cc*10)))

    # Check for input via savefile
    if root is not None:
        # Create savefile with name as outlined in docstring
        try:
            os.mkdir(libdir)
        except:
            holding = 1
        fname = "{a}_{b}x{c}_{d}_{e}_{f}_{g}_{h}.csv".format(a=root.strip(), b=binx, c=biny, d=int(inrad*1000),
                                                             e=int(ourad*1000), f=rrrr, g=ssss, h=ccc)
        fname = libdir + fname
        try:  # Look for if the file is already existing
            img = np.loadtxt(fname)
            if verbose:
                print('Reading image from {}'.format(fname))
            return img
        except:
            wrtsavfil = True
    else:
        wrtsavfil = False

    # Define the output
    xc, yc = 0, 0
    img = 0.0*ccircle(xc, yc, inrad, xrange=xr, yrange=yr, nx=binx, ny=biny, norm=norm, small=small,
                      rand=rand)
    # Early exit check
    if early:
        return img

    # Project each pixel
    kx, ky = int(binx/2), int(biny/2)  # Middle pixel coordinates
#    pixarea = (xpix * RSun) * (ypix * RSun)  # Calculate pixel area in units of RSun(?) commented out as pixarea not used in original
    # iterate over x up to midpoint
    for ix in tqdm.tqdm(range(kx)):
        for iy in (range(ky)):  # iterate over y
        #    d = np.sqrt(((xpix**2)*((ix - (binx/2))**2)) + ((ypix**2)*(iy - (biny/2))**2))  # Commented out in original
            dx, dy = xpix * (ix - binx/2), ypix * (iy - biny/2)
            # Project the path
            p = pathproj([dx, dy], inrad, maxrad=ourad, verbose=verbose, rfunct=rfunct, sclht=sss, nstep=nstep,
                         coneang=coneang)
            if p < 0:
                print('error at ix = {}, iy = {}. p < 0'.format(ix, iy))
                return img, ix, iy
            # Symmetric so replicate in each quarter
            img[ix, iy] = p
            img[binx-ix-1, iy] = p
            img[ix, biny-iy-1] = p
            img[binx-ix-1, biny-iy-1] = p

    # Save the image
    if wrtsavfil:
        if verbose > 0:
            print('Writing image to {}'.format(fname))
            plt.figure()
            plt.imshow(img)
            plt.colorbar()
            plt.title(fname[:-4])
            plt.tight_layout()
        plt.savefig('{}.png'.format(fname[:-4]))
        plt.show()
        np.savetxt(fname, img)

    # Return image
    return img
