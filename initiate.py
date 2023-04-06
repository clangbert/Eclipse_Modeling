import os
from glob import glob

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
per = 2.21857312 * day
amu = 1.166054e-24  # grams

def initiate(height, dense, abund):
    # Create the directories for saving
    try :
        os.mkdir('../../exoXtransit/')
    except:
        holding = 1
    try :
        os.mkdir('../../exoXtransit/plots')
    except:
        holding = 1
    try :
        os.mkdir('../../exoXtransit/sphrojfils')
    except:
        holding = 1
    try :
        os.mkdir('../../exoXtransit/sphrojcolfils')
    except:
        holding = 1

    try :
        os.mkdir('../../exoXtransit/absorb_spec_img')
    except:
        holding = 1
    try :
        os.mkdir('../../exoXtransit/eclipse_panes')
    except:
        holding = 1
    try :
        os.mkdir('../../exoXtransit/eclipse_panes/2ComponentBetter')
    except:
        holding = 1
    try :
        os.mkdir('../../exoXtransit/eclipse_panes/data')
    except:
        holding = 1
    try:
        os.mkdir('../../exoXtransit/eclipse_panes/data/sclht{}'.format(round(height, 5)))
    except:
        print('../../exoXtransit/eclipse_panes/data/sclht{} already exists'.format(round(height, 5)))
    try:
        os.mkdir('../../exoXtransit/eclipse_panes/data/sclht{}/dense{}'.format(round(height, 5), dense / (1e+10)))
    except:
        holding = 1
    try:
        os.mkdir(
            '../../exoXtransit/eclipse_panes/data/sclht{}/dense{}/abund{}'.format(round(height, 5), dense / (1e+10), abund))
    except:
        holding = 1

    try:
        os.mkdir('../../exoXtransit/eclipse_panes/2ComponentBetter/sclht{}'.format(round(height, 5)))
    except:
        print('../../exoXtransit/eclipse_panes/2ComponentBetter/sclht{} already exists'.format(round(height, 5)))
    try:
        os.mkdir('../../exoXtransit/eclipse_panes/2ComponentBetter/sclht{}/dense{}'.format(round(height, 5), dense / (1e+10)))
    except:
        holding = 1
    try:
        os.mkdir(
            '../../exoXtransit/eclipse_panes/2ComponentBetter/sclht{}/dense{}/abund{}'.format(round(height, 5), dense / (1e+10),
                                                                            abund))
    except:
        holding = 1

    try:
        os.mkdir('../../exoXtransit/sphrojcolfils/sclht{}'.format(round(height, 5)))
    except:
        print('../../exoXtransit/sphrojcolfils/sclht{} already exists'.format(round(height, 5)))
    try:
        os.mkdir('../../exoXtransit/sphrojcolfils/sclht{}/dense{}'.format(round(height, 5), dense / (1e+10)))
    except:
        holding = 1
    try:
        os.mkdir('../../exoXtransit/sphrojcolfils/sclht{}/dense{}/abund{}'.format(round(height, 5), dense / (1e+10), abund))
    except:
        holding = 1

    try:
        os.mkdir('../../exoXtransit/absorb_spec_img/sclht{}'.format(round(height, 5)))
    except:
        print('../../exoXtransit/absorb_spec_img/sclht{} already exists'.format(round(height, 5)))
    try:
        os.mkdir('../../exoXtransit/absorb_spec_img/sclht{}/dense{}'.format(round(height, 5), dense / (1e+10)))
    except:
        holding = 1
    try:
        os.mkdir('../../exoXtransit/absorb_spec_img/sclht{}/dense{}/abund{}'.format(round(height, 5), dense / (1e+10), abund))
    except:
        holding = 1
