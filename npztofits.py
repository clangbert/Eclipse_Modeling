import numpy as np
import matplotlib.pyplot as pyplt
from astropy.io import fits
from astropy.table import Table
import glob
from sherpa.astro.ui import *
from pycrates import read_file


tab = read_file('merged_ltc_gehrels.fits')
pha = tab.get_column('phase').values-0.5
rate = tab.get_column('counts').values/(.005*2.21857312*86400*5)
#erate = tab.get_column('net_err').values/(.005*2.21857312*86400*5)
erate = np.sqrt(rate)/((pha[-1]-pha[0])*.005*2.21857312*86400*5)
xspec = np.loadtxt('emissionspec_corkT0.6312_ism.csv')[0]

tab = read_file('merged_ltc_soft.fits')
pha = tab.get_column('phase').values-0.5
softrate = tab.get_column('counts').values/(.005*2.21857312*86400*5)
softerate = np.sqrt(softrate)/((pha[-1]-pha[0])*.005*2.21857312*86400*5)
#xspec = np.loadtxt('emissionspec_corkT0.7002_ism.csv')[0]
print(rate)
tab = read_file('merged_ltc_hard.fits')
pha = tab.get_column('phase').values-0.5
hardrate = tab.get_column('counts').values/(.005*2.21857312*86400*5)
harderate = np.sqrt(hardrate)/((pha[-1]-pha[0])*.005*2.21857312*86400*5)
#xspec = np.loadtxt('emissionspec_corkT0.7002_ism.csv')[0]
print(harderate, hardrate)

paths = [glob.glob('*/*/sclht0.01/dense89.0/abund11*.npz')]
paths.append(glob.glob('*/*/sclht0.0502/dense89.0/abund11*.npz'))
paths.append(glob.glob('*/*/sclht0.0702/dense1.0/abund11*.npz'))

tarr = np.linspace(0.455, 0.545, 201)
j=0

for pset in paths:
    filnamsplit = pset[1].split('/')
    filname = f'{filnamsplit[2]}_{filnamsplit[3]}_{filnamsplit[4][:7]}.fits'
    with np.load(pset[0]) as splot:
        splot_spec = splot['spec'][:48]
        splot_cts = splot['cts']
        splot_avg = splot['avg']
        splot_comp = splot['comp']
    with np.load(pset[1]) as count:
        count_spec = count['spec'][:48]
        count_cts = count['cts']
        count_avg = count['avg']
        count_comp = count['comp']

    xsplot = np.linspace(0.24, 7.007, len(splot_spec[0][0]))
    xcount = np.linspace(0.24, 7.007, len(count_spec[0][0]))

    softmaskcts = np.minimum((xcount>0.5), (xcount<0.9))
    hardmaskcts = np.minimum((xcount>0.9), (xcount<7))
    softmasksrc = np.minimum((xsplot>0.5), (xsplot<0.9))
    hardmasksrc = np.minimum((xsplot>0.9), (xsplot<7))

    soft_count = np.insert(np.sum(count_spec[:48,:,softmaskcts], axis=-1), 0, np.sum(np.sum(count_spec[:48,:,softmaskcts], axis=-1), axis=0)/48, axis=0)
    hard_count = np.insert(np.sum(count_spec[:48,:,hardmaskcts], axis=-1), 0, np.sum(np.sum(count_spec[:48,:,hardmaskcts], axis=-1), axis=0)/48, axis=0)

    soft_splot = np.insert(np.sum(splot_spec[:48,:,softmasksrc], axis=-1), 0, np.sum(np.sum(splot_spec[:48,:,softmasksrc], axis=-1), axis=0)/48, axis=0)
    hard_splot = np.insert(np.sum(splot_spec[:48,:,softmasksrc], axis=-1), 0, np.sum(np.sum(splot_spec[:48,:,softmasksrc], axis=-1), axis=0)/48, axis=0)

    print(np.shape(splot_spec))

    broad_names = np.insert([f'AIA_{i}' for i in range(48)], 0, 'AVG')
    broad_names = np.insert(broad_names, 0, 'PHASE')
    print(broad_names)
    print(np.shape(count_cts[:48]))

    count_cts = np.insert(count_cts[:48], 0, count_avg, axis=0)
    broad_counts = np.insert(count_cts, 0, tarr, axis=0)
    broad_counts_table = fits.BinTableHDU(Table(np.transpose(broad_counts), names=broad_names), name='Broad Counts')

    splot_cts = np.insert(splot_cts[:48], 0, splot_avg, axis=0)
    broad_splots = np.insert(splot_cts, 0, tarr, axis=0)
    broad_splots_table = fits.BinTableHDU(Table(np.transpose(broad_splots), names=broad_names), name='Broad Source')

    soft_counts = np.insert(soft_count, 0, tarr, axis=0)
    soft_counts_table = fits.BinTableHDU(Table(np.transpose(soft_counts), names=broad_names), name='Soft Counts')
    soft_splots = np.insert(soft_splot, 0, tarr, axis=0)
    soft_splots_table = fits.BinTableHDU(Table(np.transpose(soft_splots), names=broad_names), name='Soft Source')

    hard_counts = np.insert(hard_count, 0, tarr, axis=0)
    hard_counts_table = fits.BinTableHDU(Table(np.transpose(hard_counts), names=broad_names), name='Hard Counts')
    hard_splots = np.insert(hard_splot, 0, tarr, axis=0)
    hard_splots_table = fits.BinTableHDU(Table(np.transpose(hard_splots), names=broad_names), name='Hard Source')

    source_spec_tab = fits.PrimaryHDU(splot_spec)#, header=
    source_spec_tab.header['INFO'] = f'''Shape is {np.shape(splot_spec)}, reffering to 48 AIA images, 201 timesteps (see table:TGRID), and 676 spectral bins (see table:XSOURCESPEC).'''

    count_spec_tab = fits.ImageHDU(count_spec)#, header=
    source_spec_tab.header['INFO'] = f'''Shape is {np.shape(count_spec)}, reffering to {np.shape(count_spec)[0]} AIA images, {np.shape(count_spec)[1]} timesteps (see table:PHASEGRID), and {np.shape(count_spec)[-1]} spectral bins (see table:XCOUNTSPEC).'''
    print(np.shape([tarr]))
    TGRID = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='PHASEGRID')
    print(np.shape([xsplot]))
    XSOURCESPEC = fits.BinTableHDU(Table([xsplot], names=['Energy (keV)']), name='XSOURCESPEC')
    print(np.shape([xcount]))
    XCOUNTSPEC = fits.BinTableHDU(Table([xcount], names=['Energy (keV)']), name='XCOUNTSPEC')

    hdul = fits.HDUList([source_spec_tab, count_spec_tab, TGRID, XSOURCESPEC, XCOUNTSPEC,
                         broad_counts_table, broad_splots_table, soft_counts_table,
                         soft_splots_table, hard_counts_table, hard_splots_table])

    #hdul.writeto(filname)
    #hdul.close()
    j += 1


print(paths)
