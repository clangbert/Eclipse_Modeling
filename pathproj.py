import numpy as np
import matplotlib.pyplot as plt
import math as ma
import tqdm
from matplotlib.colors import LogNorm


def pathproj(offset, minrad=0, rfunct='', sclht=None, maxrad=None, nstep=1001, halfsh=None, coneang=None,
             verbose=False):
    '''
    Function:   pathproj
                projects intensity along line of sight (LOS) through stellar atmosphere onto a plane

    Parameters:
    :param offset:  Offset of the LOS from the centre (list)
    :param minrad:  Radius of opaque sphere (defaults to 0)
    :kwag rfunct:   Function describing intensity variation along radius. Options:
                        'const' ==> constant (the default)
                        'exp' ==> Exponential drop off (see sclht)
                        'invsq' ==> Inverse square
    :kwag sclht:    Scale height for exponential intensity drop off
    :kwag maxrad:   Maximum distance to integrate to, default is twice min rad or 1, whichever is greater
    :kwag nstep:    Number of steps along LOS (defaults to 1001, if < 0 then the steps are logarithmic)
    :kwag halfsh:   If set, does not double up past the edge of opaque inner sphere
    :kwag coneang:  Confines integral over the LOS to be within (+ve) or without (-ve) a given cone. Must be in degrees
                    between -90 and 90.
    :kwag verbose:  Controls chatter.
    :return:

    History:
        [27/09/2021]    Python file created derived by Joseph Hall from pathproj.pro by V. Kashyap. Attempting to make
                        this work and (somehow) learn IDL simultaneously. Maybe not my best idea...
                        Okay somehow I think this actually works
        [28/09/2021]    Added test image to see if the thing actually works and, would you believe it, it does!
    '''
    # Define maxrad
    if maxrad is None:
        maxrad = minrad * 2 if minrad >= 0.5 else 1
    # Measure length of offset list[
    try: noff = len(offset)
    except:
        print('Error, parameter offset as provided is not a list.')
        return 0.
    if type(offset) == str:
        print('Error, parameter offset as provided is not a list.')
        return 0.

    # Define variables
    d = 0; r0 = 0.; r1 = 2 * r0 if r0 >= 0.5 else 1; h = (r1 - r0) if (r1-r0) >= 1 else 1; funct = 'const'
    dl = (r1 - r0)/(100. - 1.); dlog = 0; nlos = 1001; thetamin = 0; thetamax = 90
    dx = offset[0]; dy = offset[1] if noff > 1 else 0; dd = (dx**2 + dy**2)**0.5

    try:
        r0 = minrad[0] if len(minrad) != 0 else r0
    except:
        r0 = minrad

    try:
        r1 = maxrad[0] if len(maxrad) != 0 else r1
    except:
        r1 = maxrad

    if sclht is not None:
        h = sclht

    if len(rfunct) > 0:
        funct = rfunct

    if nstep is not None:
        nlos = nstep
    if nlos < 0:
        dlog = 1; nlos = -nlos

    if coneang is not None:
        if coneang > 0:
            thetamax = coneang % 90
            r1 = dd / (np.cos(thetamax * np.pi / 180))  # Commented out in the original (I think)
        else:
            thetamin = np.abs(coneang) % 90
            r0 = dd / (np.cos(thetamin * np.pi / 180))  # Commented out in the original (I think)

    # Calculate points along the LOS
    if dd > r1:
        return 0  # 'ddr1'

    l0 = (r0**2 - dd**2)**0.5 if r0 >= dd else 0
    l1 = (r1**2 - dd**2)**0.5 if r1 >= dd else 0
    if l1 <= l0:
        return 0  # 'l0l1'

    if dlog == 1:
        lmax = np.log(l1)
        lmin = np.log(l0) if l0 > 0 else lmax - 6
        dl = (lmax-lmin)/np.abs(nlos)
        los = np.linspace(lmin, lmax, np.abs(nlos)+1)
    else:
        dl = (l1 - l0)/nlos
        los = np.linspace(l0, l1, nlos+1)

    ll = los[:-1] + 1
    dlos = los[1:] - los[:-1]
    # Translate LOS into r units and get theta
    rr = np.sqrt(los**2 + dd**2)
    drr = abs(rr[1:]-rr[:-1])
    r = 0.5*(rr[:-1]+rr[:1])
    #print(rr[0], r[0])
    xx = 0*los + dx
    yy = 0*los + dy
    # print(np.sqrt(xx**2 + los**2)/np.sqrt(xx**2 + yy**2 + los**2))
    #return
    theta = np.arccos(np.sqrt(xx**2 + los**2)/np.sqrt(xx**2 + yy**2 + los**2))*180/np.pi
    # print(theta)

    ok = []
    for index, val in enumerate(theta):
        if thetamin <= val <= thetamax:
            ok.append(index)
    mok = len(ok)
    if mok == 0:
        return 0  # 'dull', theta  # because not in an interesting area



    # Define the intenisty function
    if funct == 'invsq':  # Inverse square
        ff = 0*r
        oo = []
        for index, val in enumerate(r):
            if val > r0:
                oo.append(index)
        moo = len(oo)
        if moo > 0:
            ff[oo] = 1./r[oo]**2
    elif funct == 'exp':  # Exponential drop off
        ff = np.exp(-(r-r0)/h)
    #elif funct == 'exp2':  # second gen exponential
        #h = 9.81 * 1
        #ff = np.exp(-(r-r0)
    else:
        ff = 0*r + 1  # Constant and default


    # Build up the integrand
    # pp = ff * r * drr / ll  # commented out in the original code
    # print(dlos)
    ok = np.array(ok)
    pp = ff[ok-1] * dlos[ok-1]

    # And the integral
    p = np.sum(pp)
    # print(p)
    # Half shell transparency
    if halfsh is None and dd >= r0:
        p *= 2
    # print(pp.std())
    if verbose > 50:
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(los[:len(pp)], pp)
        axes[0].set_xlabel('Line of Sight')
        axes[1].plot(rr[:len(pp)], pp)
        axes[1].set_xlabel('Radius')
        fig.suptitle('LOS_RAD_PLOT_min{}_max{}_off{}_func{}.png'.format(r0, r1, offset, rfunct))
        fig.tight_layout()
        fig.savefig('LOS_RAD_PLOT_min{}_max{}_off{}_func{}.png'.format(r0, r1, offset, rfunct))
        fig.show()

    return p

'''sample_image = np.zeros([1000, 1000])
for rowind, row in tqdm.tqdm(enumerate(sample_image)):
    for colind, col in enumerate(row):
        sample_image[rowind, colind] = pathproj([colind-500, rowind-500], minrad=1, maxrad=250, rfunct='invsq', verbose=9, halfsh=1, nstep=1001)

plt.imshow(sample_image+.000001, norm=LogNorm())
plt.colorbar()
plt.title('Sample integral of path proj for 1000x1000 array with a \ncentred source of max radius 250')
plt.savefig('Sample star test.png')
plt.show()'''
