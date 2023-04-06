import numpy as np
import matplotlib.pyplot as pyplt
from astropy.io import fits
from astropy.table import Table
import glob
from sherpa.astro.ui import *


def fits_gen(xcounts, xsource, naia, tarr, aias, src_spec_img, cts_spec_img, path):
    srccts = ['src', 'count']
    source_col = fits.ColDefs([fits.Column(name='Energy', format='D', array=xsource)])
    count_col = fits.ColDefs([fits.Column(name='Energy', format='D', array=xcounts)])
    cols = [source_col, count_col]

    AIA_tab = fits.PrimaryHDU(aias)#, name='AIA_imgs')
    SRC_IMG = fits.ImageHDU(src_spec_img, name='SRC_IMG')
    CTS_IMG = fits.ImageHDU(cts_spec_img, name='CTS_IMG')
    PHA_GRID = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='PHA_GRID')
    BROAD_SRC = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='BROAD_SRC')
    USOFT_SRC = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='USOFT_SRC')
    SOFT_SRC = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='SOFT_SRC')
    MED_SRC = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='MED_SRC')
    HARD_SRC = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='HARD_SRC')
    BROAD_CTS = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='BROAD_CTS')
    USOFT_CTS = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='USOFT_CTS')
    SOFT_CTS = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='SOFT_CTS')
    MED_CTS = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='MED_CTS')
    HARD_CTS = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='HARD_CTS')

    AVG_SRC = fits.BinTableHDU.from_columns(source_col, name=f'AVG_SRC_SPEC')
    AVG_CTS = fits.BinTableHDU.from_columns(count_col, name=f'AVG_CTS_SPEC')

    tablist = [AIA_tab, PHA_GRID, BROAD_SRC, USOFT_SRC, SOFT_SRC,
                MED_SRC, HARD_SRC, BROAD_CTS, USOFT_CTS, SOFT_CTS, MED_CTS, HARD_CTS]
               #AVG_SRC, AVG_CTS] #SRC_IMG, CTS_IMG,
    for i in range(naia):
        for jind, j in enumerate(srccts):
            for comp in [0, 1]:
                #print(type(cols[jind]))
                AIA_Tab = fits.BinTableHDU.from_columns(cols[jind], name=f'AIA{i}_comp{comp+1}_{j}')
                tablist.append(AIA_Tab)
    #print('tab list made')
    hdul = fits.HDUList(tablist)
    #print(hdul.info())
    hdul.writeto(path, overwrite=True)
    #print('written')
    hdul.close()
    #print('closed')

def fits_gen_lite(xcounts, xsource, naia, tarr, aias, src_spec_img, cts_spec_img, path):
    srccts = ['src', 'count']
    source_col = fits.ColDefs([fits.Column(name='Energy', format='D', array=xsource)])
    count_col = fits.ColDefs([fits.Column(name='Energy', format='D', array=xcounts)])
    cols = [source_col, count_col]

    AIA_tab = fits.PrimaryHDU(aias)#, name='AIA_imgs')
    #SRC_IMG = fits.ImageHDU(src_spec_img, name='SRC_IMG')
    #CTS_IMG = fits.ImageHDU(cts_spec_img, name='CTS_IMG')
    PHA_GRID = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='PHA_GRID')
    BROAD_SRC = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='BROAD_SRC')
    # USOFT_SRC = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='USOFT_SRC')
    # SOFT_SRC = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='SOFT_SRC')
    MED_SRC = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='MED_SRC')
    # HARD_SRC = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='HARD_SRC')
    BROAD_CTS = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='BROAD_CTS')
    # USOFT_CTS = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='USOFT_CTS')
    # SOFT_CTS = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='SOFT_CTS')
    MED_CTS = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='MED_CTS')
    # HARD_CTS = fits.BinTableHDU(Table(np.transpose([tarr]), names=['PHASE']), name='HARD_CTS')

    AVG_SRC = fits.BinTableHDU.from_columns(source_col, name=f'AVG_SRC_SPEC')
    AVG_CTS = fits.BinTableHDU.from_columns(count_col, name=f'AVG_CTS_SPEC')

    # tablist = [AIA_tab, PHA_GRID, BROAD_SRC, USOFT_SRC, SOFT_SRC,
    #            MED_SRC, HARD_SRC, BROAD_CTS, USOFT_CTS, SOFT_CTS, MED_CTS, HARD_CTS]
    tablist = [AIA_tab, PHA_GRID, BROAD_SRC, MED_SRC, AVG_SRC, BROAD_CTS, MED_CTS, AVG_CTS]
               #AVG_SRC, AVG_CTS] #SRC_IMG, CTS_IMG,
    for i in range(naia):
        for jind, j in enumerate(srccts):
            for comp in [0, 1]:
                #print(type(cols[jind]))
                AIA_Tab = fits.BinTableHDU.from_columns(cols[jind], name=f'AIA{i}_comp{comp+1}_{j}')
                tablist.append(AIA_Tab)
    #print('tab list made')
    hdul = fits.HDUList(tablist)
    #print(hdul.info())
    hdul.writeto(path, overwrite=True)
    #print('written')
    hdul.close()
    #print('closed')
