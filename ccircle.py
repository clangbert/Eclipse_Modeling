import random
import numpy as np
import matplotlib.pyplot as plt
import math as ma
import tqdm


def ccircle(xc, yc=None, rad=None, xrange=[1, 512], yrange=[1, 512], nx=512, ny=512, norm=False, small=False, rand=False,
            verbose=False):
    """
    Function:
        ccircle, returns a bit image of a circle
    Syntax:
        img = ccircle(xc, yc, xrange=xrange, yrange=yrange, nx=nx, ny=ny, norm=norm,
                      small=small , rand=rand)

    Parameters:
        :param xc:      x-coordinate of the centre
        :param yc:      y-coordinate of the centre, if not given then assumed to be same as xc
        :param rad:     Radius of circle, if ONLY xc given then assumed as 10.
                        If only xc and yc given, yc set to be xc
                        and rad takes on the second parameter
        :key xrange:    Range of x values (defaults to [1, 512])
        :key yrange:    Range of y values (defaults to [1, 512])
        :key nx:        No. elements in x (default is 512)
        :key ny:        No. elements in y (default is 512)
        :key norm:      Boolean, False by default.
                        Tells function whether or not to normalise image.
        :key small:     Boolean, False by default.
                        Stores image of smaller size
        :key rand:      Boolean, False by default.
                        If True resorts to monte-carlo trick to speed things up.
        :key verbose:   Boolean, False by default.
                        Controls chatter, if True, will plot image with stats.
    :return: A bit image of a circle

    History:
        modified from imgcreate.pro by VK	{8/12/93}
        added keyword _EXTRA (VK; Apr99)
        [28/09/2021]    Python file created. Adapted from V. Kashyap's ccircle.pro by Joseph Hall.
    """
    # set parameters per the docstring
    if yc is None:
        yc = xc
        if rad is None:
            rad = 10
    elif yc is not None and rad is None:
        rad = abs(yc)
        yc = xc
    
    # Catch errors for incorrect type
#    if type(xrange) == (int or float):
 #   	xrange = [1, xrange]
  #  if type(yrange) == (int or float):
   # 	yra = [1, yra]

    # Set up arrays
    x = np.linspace(xrange[0], xrange[1], nx)
    y = np.linspace(yrange[0], yrange[1], ny)
    xstp, ystp = (x[-1] - x[0])/(len(x)-1), (y[-1] - y[0])/(len(y)-1)  # x and y step sizes

    img = np.zeros((nx, ny))  # image array
    
    # Set up the corners
    x0, x1 = xc - rad - xstp, xc + rad + xstp
    x0 = x0 if x0 < xrange[1] else xrange[1]; x0 = x0 if x0 > xrange[0] else xrange[0]
    x1 = x1 if x1 < xrange[1] else xrange[1]; x1 = x1 if x1 > xrange[0] else xrange[0]
    
    y0, y1 = yc - rad - ystp, yc + rad + ystp
    y0 = y0 if y0 < yrange[1] else yrange[1]; y0 = y0 if y0 > yrange[0] else yrange[0]
    y1 = y1 if y1 < yrange[1] else yrange[1]; y1 = y1 if y1 > yrange[0] else yrange[0]

    # Find corner indicies
    # X minimum first
    h1 = (x < x0).nonzero()
    i0 = h1[0][-1] if len(h1[0]) > 0 else 0
    # X max
    h1 = (x > x1).nonzero()
    i1 = h1[0][0] if len(h1[0]) > 0 else nx-1
    # Y minimum
    h1 = (y < y0).nonzero()
    j0 = h1[0][-1] if len(h1[0]) > 0 else 0
    # y max
    h1 = (y > y1).nonzero()
    j1 = h1[0][0] if len(h1[0]) > 0 else ny - 1

    # Set up random point scatter if circle is small
    di, dj = i1 - i0 + 1, j1 - j0 + 1
    npt = di * dj
    if rand is not False:
        npt = npt*10
        xpt = (x[i1] - x[i0] + 1)*np.random.uniform(size=npt) + x[i0] - 0.5
        ypt = (y[j1] - y[j0] + 1)*np.random.uniform(size=npt) + y[j0] - 0.5
        rpt = (xpt-xc)**2 + (ypt-yc)**2
        h1 = (rpt <= rad**2).nonzero()
        ix, iy = np.fix(xpt + 0.5).astype('int32'), np.fix(ypt + 0.5).astype('int32')
        if len(h1[0]) != 0:
            for i in h1[0]:
                img[ix[i]][iy[i]] += 1
    # And for the non-random case
    else:
        z = np.arange(j1 - j0 + 1) + j0
        for i in range(i0, i1):
            tmp = (x[i] - xc)**2 + (y[j0:j1] - yc)**2
            h1 = (tmp <= rad**2).nonzero()
            if len(h1[0]) != 0:
                img[i, z[h1]] = 1

    # see if need to return smaller image
    if small is True:
        img = img[i0:i1, j0:j1]  # Constrained by the corners

    # Normalise the image (if relevant)
    if norm is True:
        nconst = img.max() if img.max() != 0 else 1
        img /= nconst

    # Check verbosity and plot
    if verbose > 0:
        plt.imshow(img)
        plt.title(
            'CIRCLE(xc={}, yc={}, rad={}, rand={}, small={}, norm={})'.format(round(xc, 2), round(yc, 2), round(rad, 2),
                                                                              rand, small, norm))
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(
            'CIRCLE(xc={}, yc={}, rad={}, rand={}, small={}, norm={}).png'.format(round(xc, 2), round(yc, 2),
                                                                                  round(rad, 2), rand, small, norm))
        plt.show()

    return img


'''c = ccircle(xc=240, yc=340, rad=50, rand=True, small=True, verbose=True, norm=True)'''
