import glymur
from astropy.io import fits
import glob
import numpy as np

fils = glob.glob('*.jp2')

for fil in fils:
    img = glymur.Jp2k(fil)[:]
    fits_img = fits.PrimaryHDU(img)
    hdul = fits.HDUList([fits_img])
    print(hdul.info())
    hdul.writeto(fil.split('.')[0]+'.fits', overwrite=True)
    hdul.close
