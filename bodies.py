import numpy as np
from sphroj import sphroj
import mendeleev
from sherpa.astro.ui import *
from AIAFile import *

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
per = 2.21857312*day
amu = 1.166054e-24  # grams
pc = 3.086e+18  # cm

def distshift(nx, ny=False):
    """
    Recreates line 244 (shift(dist(nx,ny),nx/2,ny/2)) in the original mk_bands_tau.pro by Vinay Kashyap. Adapted from:
    https://gist.github.com/abeelen/453de325dd9787ea2aa7fad495f4f018
    :param nx: Number of columns
    :param ny: Number of rows (defaults to match number of columns)
    :return: A rectangular array in which the value of each element is proportional to its frequency.
    """
    # Set up the variables
    ny = nx if ny is False else ny
    nay = ny
    nax = nx
    # Check if more rows than columns -- if that is the case, flip the nx and ny variables, transposes back into
    # original shape at the end
    if ny > nx:
        nx, ny = nay, nax
    # Set up the original axis, checking if number of cols is even
    if nx % 2 == 0:
        ax = np.linspace(int(-nx/2)+1, int(nx/2), nx)
    else:
        ax = np.linspace(int(-nx / 2), int(nx / 2), nx)
    res = np.sqrt(ax[:, np.newaxis] ** 2 + ax ** 2)  # Convert to an nx by nx square array
    test = np.zeros_like(res)  # Blank array matching shape of res
    # Need to flip the res array so that it matches the result of the IDL function, iterate over all values and move
    # them to the opposite side of the test array
    for rowind, row in enumerate(res):
        for colind, val in enumerate(row):
            test[-(rowind + 1), -(colind + 1)] = val
    # Constrain for number of rows
    if ny % 2 != 0 and nx % 2 == 0:
        # Specific case where columns even but rows odd -- needs the addition of 1 to each index to produce same result
        # as IDL original
        final = test[int((nx - ny) / 2) + 1:int((nx + ny) / 2) + 1]
    else:
        final = test[int((nx - ny) / 2):int((nx + ny) / 2)]
    # Check if need to transpose to return the desired shape
    if nay <= nax:
        return final
    else:
        final = np.transpose(final)
        return final


# Set up classes for star and planet
class Star:
    def __init__(self, objnam='HD189733A', mstar=0.823, rstar=0.788, corkT1=0.201, corkT2=0.646, fudgeht=1.0,
                 per=2.21857312,
                 xrayfx=1e-13, distance=19.77, coneang=50, nx=512, ny=512):

        """
        Initialise the star object, all values default to those of HD189733A
        :param objnam: name of the object to be appended to save files, defaults to 'HD189733A'
        :param mstar: Mass of the star (solar units)
        :param rstar: Radius of the star (solar units)
        :param corkT: Temperature of the corona in KeV
        :param fudgeht: Fudge factor accounting for stellar corona extending to a greater than expected sclht
        :param per: orbital period of the planet (days)
        :param xrayfx: Base xray flux count rate.
        """
        self.name = objnam
        self.mass = mstar
        self.rad = rstar
        self.corkT1 = corkT1
        self.corT1 = (corkT1 * 1e+3) / degev # temperature in K
        self.corkT2 = corkT2
        self.corT2 = (corkT2 * 1e+3) / degev
        self.per = per  # * day
        self.accg = grav * (self.mass * msun) / ((self.rad * rsun) ** 2)
        self.abund = self.set_abund()
        elements = mendeleev.get_all_elements()
        elmass = np.array([i.atomic_weight for i in elements[:len(self.abund)]])
        mmm = (sum(self.abund * elmass) / sum(self.abund)) * amu
        self.coronaht1 = fudgeht * (kB * self.corT1 / mmm / self.accg / rsun)
        self.coronaht2 = fudgeht * (kB * self.corT2 / mmm / self.accg / rsun)
        print('corona heights {} and {}'.format(self.coronaht1, self.coronaht2))
        self.fudgeht = fudgeht
        self.xrayfx = xrayfx
        # Setting up ctrt -- depends on a file i dont have 'cecf_ea12_grid.save' so will use the commented out value for
        # now of 0.05
        self.ctrt = 0.05
        # Version for when i have the file (left in IDL format for now)
        # ctrt=interpol(reform(eccf[*,0,0]),logTarr,alog10(corT))*xrayfx
        self.dist = distance * pc
        self.conecc = coneang

        ## Generate images for each component of the star
        self.solimg1 = np.zeros((nx,
                                 ny))  # self.get_star_img(rstar, self.coronaht1, vis, xra, yra, nx, ny, rfunct, conecc, self.corkT1, useaia=useaia)
        self.solimg2 = np.zeros(
            (nx, ny))  # self.get_star_img(r1, h3, vis, xra, yra, nx, ny, rfunct, conecc, self.corkT2, useaia=useaia)

        self.solimgarr = np.array([self.solimg1, self.solimg2])
        print('here1')

    def get_star_img(self, r1, h1, vis, xra, yra, nx, ny, corkT, rfunct='exp', conecc=50, useaia=True):
        img1 = sphroj(r1, r1 + 5 * h1, verbose=vis, xrange=xra, yrange=yra, nx=nx, ny=ny, sclht=h1, rfunct=rfunct,
                      coneang=conecc,
                      root=self.name + '_corkT' + str(round(corkT, 4)), libdir='../../exoXtransit/sphrojfils')
        img1 = img1 / np.sum(img1)
        nfil = 1
        solimg = np.zeros([1, nx, ny])
        solimg[0] = img1
        solnorm = np.zeros(nfil) + np.sum(img1)
        dimg = distshift(nx, ny)
        dd = np.unique(dimg)
        ndd = len(dd)
        rimg = np.zeros(ndd)
        rmx = np.max(rimg)

        #   for ir in tqdm.tqdm(range(ndd)):
        #        od = (dimg == dd[ir])
        #         rimg[ir] = np.mean(img1[od])
        #      rmx = np.max(rimg)
        #       oir = (rimg > 0).nonzero()
        #        moir = len(oir)

        # Get AIA images
        if useaia is True:
            print(self.name + str(round(corkT, 4)))
            aiasavfil = f'../../exoXtransit/norm_useAIA_{self.name}_{nx}x{ny}_cone{conecc}_corht{round(h1, 4)}_corkT{round(corkT, 4)}.pk1'
            AIAims = get_aia(aiasavfil, nx, ny, r1, img1, xra, ndd, rimg, dimg, dd, rmx, img1,
                             self.name + str(round(corkT, 4)))  # Load AIA image data
            solimg = AIAims.solimg
            solnorm = AIAims.solnorm
            nfil = AIAims.nfil
            # plt.imshow(solimg[10])
            # plt.show()
        return solimg

    def set_abund(self):
        abund = [1.00E+00, 8.51E-02, 1.26E-11, 2.51E-11, 3.55E-10, 3.31E-04, 8.32E-05, 6.76E-04,
                 3.63E-08, 1.20E-04, 2.14E-06, 3.80E-05, 2.95E-06, 3.55E-05, 2.82E-07, 2.14E-05,
                 3.16E-07, 2.51E-06, 1.32E-07, 2.29E-06, 1.48E-09, 1.05E-07, 1.00E-08, 4.68E-07,
                 2.45E-07, 3.16E-05, 8.32E-08, 1.78E-06, 1.62E-08, 3.98E-08]
        # Recast stellar elemental abundances using values produced by the fitting
        with open('mg_tiedparams_allout_fitstats_xsvapec_cstat_v10_cts.txt') as f:
            lines = f.readlines()
            parnames = lines[5]
            parvals = lines[6]
            parnames = parnames.split('(')[1].split(')')[0].split(',')
            parvals = parvals.split('(')[1].split(')')[0].split(',')

            for index, nam in enumerate(parnames):
                nam = nam.split("'")[1]
                if nam == "mdl1.O":
                    abundO = parvals[index]
                elif nam == 'mdl1.Ne':
                    abundNe = parvals[index]
                elif nam == 'mdl1.Fe':
                    abundFe = parvals[index]
                elif nam == 'mdl1.Mg':
                    abundMg = parvals[index]
        # CNO
        abund[5], abund[6], abund[7] = [i * float(abundO) for i in [abund[5], abund[6], abund[7]]]
        # Ne
        abund[9] *= float(abundNe)
        # Mg
        abund[15], abund[13], abund[12], abund[11] = [i * float(abundMg) for i in [abund[15], abund[13], abund[12], abund[11]]]
        # Iron
        abund[25], abund[27], abund[19], abund[17] = [i * float(abundFe) for i in [abund[25], abund[27], abund[19], abund[17]]]
        return abund


class Planet:
    def __init__(self, objnam='HD189733Ab', mass=1.142, rad=1.138, sclht=0.7, nbase=2e+11, inclin=89.99, parent=Star(),
                 temp=0.01):
        """
        Initialise the planet object, all values default to HD189733Ab
        :param objnam: name of the object, defaults to 'HD189733Ab'
        :param mass: Mass of the planet (MJup)
        :param rad: Radius of the planet (RJup)
        :param sclht: Atmospheric scale height (rJup)
        :param nbase: Density at base of atmosphere (cm^-3)
        :param inclin: planetary orbit inclination (deg) (0 --> no eclipse)
        :param abundO: Oxygen abundence relative to stellar (defaults to 5)
        :param abundC: Carbon abundance relative to stellar (defaults to match abundO)
        :param abundN: Nitrogen abundance relative to stellar (defaults to match abundO)
        :param abundFe: Iron abundance relative to stellar (defaults to 1)
        :param parent: Parent, defaults to a default star
        """
        self.name = objnam
        self.mass = mass * mjup / msun  # Recast into solar mass
        self.rad = rad * rjup / rsun  # Recast into solar radius
        self.sclhtRJ = sclht
        self.sclht = round(sclht * rjup / rsun, 4)  # Recast into solar radis
        print(self.sclht, 'rsun')
        print(sclht, 'rjup')
        self.nbase = nbase
        self.inclin = inclin * np.pi / 180  # Convert to radians
        self.drad = self.sclht / 2  # Slice size
        self.parent = parent
        self.temp = temp

    def set_atm_abund(self, abundO=5, abundC=False, abundN=False, abundFe=1.0):
        self.abundO = abundO
        self.abundC = abundC if abundC is not False else abundO
        self.abundN = abundN if abundN is not False else abundO
        self.abundFe = abundFe
        # Planetary abundances array
        abund = [1.00E+00, 8.51E-02, 1.26E-11, 2.51E-11, 3.55E-10, 3.31E-04, 8.32E-05, 6.76E-04,
                 3.63E-08, 1.20E-04, 2.14E-06, 3.80E-05, 2.95E-06, 3.55E-05, 2.82E-07, 2.14E-05,
                 3.16E-07, 2.51E-06, 1.32E-07, 2.29E-06, 1.48E-09, 1.05E-07, 1.00E-08, 4.68E-07,
                 2.45E-07, 3.16E-05, 8.32E-08, 1.78E-06, 1.62E-08, 3.98E-08]
        self.pabund = abund
        self.pabund[7] *= self.abundO
        self.pabund[5] *= self.abundC
        self.pabund[6] *= self.abundN
        self.pabund[25] *= self.abundFe

    def set_sclht(self):
        acc = grav * (self.mass * msun) / ((self.rad * rsun) ** 2)
        print('acc', acc)
        # Atomic masses from mendeley (keep the structure just in case)
        elements = mendeleev.get_all_elements()
        elmass = np.array([i.atomic_weight for i in elements[:len(self.pabund)]])
        mmm = (sum(self.pabund * elmass) / sum(self.pabund)) * amu
        self.sclht = ((self.temp * 10 ** 3) * kB / degev) / (acc * mmm) / rsun  # Calc in solar units
        self.drad = self.sclht / 2
