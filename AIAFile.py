import numpy as np
import matplotlib.pyplot as plt
import tqdm
import glymur
from sherpa.astro.ui import *
import glob
from congrid import congrid
import _pickle as pickle
from astropy.io import fits
import os

class AIAFile(object):
    def __init__(self, filnam, solimg, solnorm, dimg, dd, ndd, oir, rmx, nnx, nny, szi, rmx2,
                 fils, nfil):
            self.filnam = filnam
            self.solimg = solimg
            self.solnorm = solnorm
            self.dimg = dimg
            self.dd = dd
            self.ndd = ndd
            self.oir = oir
            self.rmx = rmx
            self.nnx = nnx
            self.nny = nny
            self.szi = szi
            self.fils = fils
            self.rmx2 = rmx2
            self.nfil = nfil


def get_aia(filnam, nx, ny, r1, solimg, xra, ndd, rimg, dimg, dd, rmx, img1, name):
    """
    Loads and rescales AIA images of the sun for use in the generation of simulated lightcurves
    :param filnam:  Name of the file, should countain key details of the sim. Looks to see if rescaled AIA stuff already
                    done and used to save the final result to
    :param nx:      Number of x axis pixels
    :param ny:      No. y axis pixels
    :param r1:      Radius of the star (solar units)
    :param solimg:  The current stack of star images
    :param xra:     Range of x coords for the star image
    :param ndd:
    :param oir:
    :param dimg:
    :param dd:
    :param rmx:
    :param img1:    Principle image of the star
    :return:
    """
    try:
        with open(filnam, 'rb') as inp:  # Look to see if file with the right parameters already exists
            aiaimg = pickle.load(inp)
            print('Reading AIA file from {}'.format(filnam))
            return aiaimg
    except:
        print('failed to find aia file, generating now')
        for ir in tqdm.tqdm(range(ndd)):
            od = (dimg == dd[ir])
            rimg[ir] = np.mean(img1[od])
        rmx = np.max(rimg)
        oir = (rimg > 0).nonzero()
        moir = len(oir)

        fils = glob.glob('../AIA/*_193.fits')  # Look for the AIA files
        nfil = len(fils)

        solimg = np.zeros([nfil, nx, ny])
        solnorm = np.zeros(nfil)
        rmx2 = np.zeros(nfil)
        nstar = nx * r1 * 2. / (xra[1] - xra[0])
        scale = nstar / 3180
        for k in tqdm.tqdm(range(nfil)):
            with fits.open(fils[k]) as hdul:
                img = hdul[0].data

            solnorm[k] = np.sum(img)
            # Get image size, assuming that all the jp2 files are identical
            if k == 0:
                szi = np.shape(img)
                mx, my = szi[0], szi[1]
                nnx, nny = 2 * int((mx * scale / 2)), 2 * int(my * scale / 2)
            img2 = congrid(img, (int(nnx), int(nny)))  # Resize the AIA image to match the scale of the star
            img2 = img2 / np.sum(img2)  # Renormalise the AIA image
            img3 = np.zeros((nx, ny))  # Blank array to paste AIA into
            # Put img2 at centre of img3
            img3[int((nx - nnx) / 2):int((nx + nnx) / 2), int((ny - nny) / 2):int((ny + nny) / 2)] = img2
            try:
                os.mkdir('../../exoXtransit/rimg2/')
            except:
                holding =1
            rimg2_fname = '../../exoXtransit/rimg2/rimg2_{}_{}x{}_{}.csv'.format(name, nx, ny, k)
            try:
                rimg2 = np.loadtxt(rimg2_fname)  # See if rimg2 already calculated for this star and AIA file
            except:
                rimg2 = np.zeros(ndd)
                for jr, ir in (enumerate(oir[0])):
                    od = (dimg == dd[ir]).nonzero()
                    if len(od) > 0:
                        rimg2[ir] = np.mean(img3[od])
                np.savetxt(rimg2_fname, rimg2)
            rmx2[k] = np.max(rimg2)
            img2 = (img1 / rmx) * rmx2[
                k]  # Rescale base image to match AIA - redo, rmx2 can be calculated once and reused, dont need to redo for every corona angle
            img3 = np.maximum(img2, img3)
            img4 = img3 / np.sum(img3)  # renormalise
            solimg[k] = img4  # Append to the AIA array
            # Save the image of the star
            #plt.imshow(img3)
            #plt.savefig('../AIA/corona/AIAim_number{}_{}.pdf'.format(k, filnam))
            #plt.close()
        print('generating aiaimg')
        aiaimg = AIAFile(filnam, solimg, solnorm, dimg, dd, ndd, oir, rmx, nnx, nny, szi, rmx2, fils, nfil)
        with open(filnam, 'wb') as outp:
            print('Saving file as {}'.format(filnam))
            pickle.dump(aiaimg, outp, protocol=-1)

        return aiaimg
