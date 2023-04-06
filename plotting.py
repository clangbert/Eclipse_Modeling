import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import AIAFile
import glob
import tqdm
import _pickle as pickle
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.ndimage import zoom
from astropy.io import fits
from pycrates import read_file
import matplotlib.patches as patches
from matplotlib import ticker
from pycrates import read_file

xra = [-8.598396877570645, 8.598396877570645]
pixsize = -(xra[0]-xra[1])/512
colours = ['red', 'blue', 'orange', 'green', 'purple', 'gold', 'c', 'm', 'y', 'grey']
print(len(colours))
rsun = 6.969e+10  # Solar radius (cm)
rjup = 6.9911e+9  # Jovian radius (cm)

def cropping(img, axis, cb=False, vmax=None, cropmin=200, cropmax=312, units=rsun):
    img2 = img[cropmin:cropmax]  # np.clip(img[cropmin:cropmax], 1e0, None)
    print(vmax)
    img3 = np.transpose(img2)[cropmin:cropmax]
    img4 = np.transpose(img3)
    ext = np.array([-len(img4) * pixsize/2, len(img4) * pixsize/2,  -len(img4) * pixsize/2, len(img4) * pixsize/2])*rsun/units
    if cb:
        im = axis.imshow(img4, extent=ext, norm=LogNorm())# if vmax is None else axis.imshow(img4, extent=ext, norm=LogNorm(vmax=vmax, vmin=1e+10))
        return axis, img4, im

    axis.imshow(img4, extent=ext, norm=LogNorm(vmin=1e-5))# if vmax is None else axis.imshow(img4, extent=ext, norm=LogNorm(vmax=vmax, vmin=1e+10))
    return axis, img4


def starimg(ncomp=2):
    AIApaths = glob.glob('*.pk1')
    print(AIApaths)
    #fig, ax = plt.subplots(3, 2, figsize=(7, 10), sharex=True, sharey=True)
    summed = np.zeros((2, 512, 512))
    specs = glob.glob('emissionspec_corkT*.csv')
    img1 = np.zeros((2, 512, 512))
    img2 = np.zeros((2, 512, 512))
    for ind, path in tqdm.tqdm(enumerate(AIApaths)):
        cts = np.sum(np.loadtxt(specs[ind])[1])
        print(cts)
        with open(path, 'rb') as fil:
            aias = pickle.load(fil)
        solsum = np.zeros(len(aias.solimg))
        for index, star in enumerate(aias.solimg):
            solsum[index] = np.sum(star)
        plt.plot(solsum)
        summed[0] += aias.solimg[0]*cts
        summed[1] += aias.solimg[25]*cts
        img1[ind] = aias.solimg[0]*cts
        img2[ind] = aias.solimg[25]*cts
        temp = path.split('corkT')[-1].split('.pk1')[0]
        #ax[ind][0].set_ylabel('Corona temp: {} keV \nCoordinates, $R_\odot$'.format(temp))
    plt.show()
    return
    ax[0][0], crop1 = cropping(img1[0]*cts, ax[0][0], vmax=np.max(summed))
    ax[0][1], crop2 = cropping(img2[0]*cts, ax[0][1], vmax=np.max(summed))

    ax[1][0], crop1 = cropping(img1[1] * cts, ax[1][0], vmax=np.max(summed))
    ax[1][1], crop2 = cropping(img2[1] * cts, ax[1][1], vmax=np.max(summed))

    ax[2][0], crop3 = cropping(summed[0], ax[2][0], vmax=np.max(summed))
    ax[2][1], crop4 = cropping(summed[1], ax[2][1], vmax=np.max(summed))
    ax[2][0].set_ylabel('Sum of above images \nCoordinates, $R_\odot$'.format(temp))
    ax[2][0].set_xlabel('Coordinates, $R_\odot$')
    ax[2][1].set_xlabel('Coordinates, $R_\odot$')
    ax[0][0].set_title('AIA image 0')
    ax[0][1].set_title('AIA image 1')

    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #cb = fig.colorbar(im, cax=cbar_ax)
    fig.tight_layout()
    fig.savefig('plots/SummedImgSampNoCb.pdf')
    fig.show()


def planimg(ab, den, ht, path='sphrojcolfils', name='density map'):
    fils = glob.glob(f'img2 inc nh col.csv')#'{path}/*{ht}/*{den}/*{ab}/*again*elo0.5*ehi7*.csv') #img2 inc nh col
    print(fils[0])
    img = np.loadtxt(fils[0])#*den*5*10**10 if 'sor' not in name else np.loadtxt(fils[0])+np.loadtxt(fils[1])
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    print('cropping')
    ax, crop, im = cropping(img, ax, cb=True, cropmin=220+17, cropmax=292-17, units=rjup)
    print('cropping done')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Coordinates, $R_{Jup}$')
    ax.set_ylabel('Coordinates, $R_{Jup}$')
    ten = '{10}'
    jup = '{Jup}'
    ax.set_title(f'''Conditions: sclht = {round(ht*rsun/rjup, 1)} $R_{jup}$, $n_0 = {den*5} \\times 10^{ten}$ atoms / cm$^3$,
[O] = {ab}''')
    fig.tight_layout()
    fig.savefig(f'{path}/sclht{ht}/dense{den}/abund{ab}/cropped_map.png')
    fig.savefig(f'{path}/sclht{ht}/dense{den}/abund{ab}/cropped_map.pdf')
    fig.savefig(f'plots/{name}.png')
    fig.show()


#planimg(5.0, 2.0, 0.0502, path='absorb_spec_img', name='density_new_3')


def getobsdat(norm=True):
    tab = read_file('../../lightcurve_repro/no_flare/merged_ltc_gehrels.fits')
    print(tab)
    pha = tab.get_column('phase').values - 0.5
    erate = tab.get_column('stat_err').values / (.005 * 2.21857312 * 86400)
    rate = tab.get_column('counts').values / (.005 * 2.21857312 * 86400)

    if norm:
        rate = rate/rate[0]
        erate = erate/rate[0]

    obs = np.array([pha, rate, erate])
    plt.errorbar(obs[0], obs[1], obs[2], ls='', color='black', marker='.', label='Stacked data')
    plt.plot([0.485, 0.485], [obs[1].max()+obs[2].max(), obs[1].min()-obs[2].max()], ls='--', alpha=1, c='grey', label='Optical Transit')
    plt.plot([0.515, 0.515], [obs[1].max()+obs[2].max(), obs[1].min()-obs[2].max()], ls='--', alpha=1, c='grey')
    plt.legend()
    plt.xlabel('Orbital phase (0.5 is mid transit)')
    plt.ylabel('Count rate (counts per second)')
    plt.savefig('stackedltc no norm.pdf')
    plt.savefig('stackedltc.png')
    plt.show()

    #print(np.shape(np.array([pha, rate, erate])))
    return np.array([pha, rate, erate])
#getobsdat(norm=False)

def lightcurves(doobs, path, sclhts, denses, abunds, savfil='lightcurve_sclht_dense_abund_plot_a.pdf', bands=['0.5-7.007']):
    fig, axes = plt.subplots(len(denses), len(sclhts), sharex=True, sharey=True, figsize=(12, 7))
    print(type(axes[0][0]))
    obsdat = getobsdat()
    for htind, sclht in tqdm.tqdm(enumerate(sclhts)):
        for denseind, dense in enumerate(denses):
            for abundind, abund in enumerate(abunds):
                datpath = path+'sclht{}/dense{}/abund{}/*0.5-7.007*ltcdata.csv'.format(sclht, dense, abund)

                fillist = glob.glob(datpath)

                if len(fillist) == 0:
                    print('nan detected in', datpath)
                alldat = np.zeros(300)
                for filind, fil in enumerate(fillist):
                    dat = np.loadtxt(fil)
                    normdat = (dat[1]/dat[1][0])
                    alldat = alldat + normdat

                avgdat = alldat/len(fillist)
                if denseind == 1 and htind == 0:
                    axes[denseind][htind].plot(dat[0], avgdat, color=colours[abundind], label='abund = ' + str(abund), alpha=1)
                    if abundind == 0:
                        axes[denseind][htind].errorbar(obsdat[0], obsdat[1], obsdat[2], ls='', marker='.', label='Chandra data', color='black')

                axes[denseind][htind].plot(dat[0], avgdat, color=colours[abundind])
                axes[denseind][htind].errorbar(obsdat[0], obsdat[1], obsdat[2], ls='', marker='.', color='black')

            axes[denseind][0].set_ylabel('Base density = {} $\\times 10^{}$ atoms cm$^{}$ \nNormalsied count rate'.format(dense*5, '{10}', '{-3}'))
            axes[-1][htind].set_xlabel('Orbital phase ($0.5$ is mid transit) \nScale height = {} $R_\odot$ '.format(sclht, '{Jup}'))
    axes[1][0].legend()
    fig.suptitle('Comparing impacts of scale height, density, and abundance on lightcurve depth.')
    fig.tight_layout()
    fig.savefig(savfil)
    fig.show()
    return


def modltc(path, sclhts, denses, abunds, savfil='comb_ltc_alldense_allabund.png'):
    obsdat = getobsdat()
    for htind, sclht in tqdm.tqdm(enumerate(sclhts)):
        fig, axes = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(10, 15))
        yp = 0
        for denseind, dense in enumerate(denses):
            xp = denseind % 5
            if denseind == 5:
                yp += 1
            for abundind, abund in enumerate(abunds):
                datpath = path+'sclht{}/dense{}/abund{}/*0.5-7.007*ltcdata.csv'.format(sclht, dense, abund)

                fillist = glob.glob(datpath)

                if len(fillist) == 0:
                    print('nan detected in', datpath)
                alldat = np.zeros(300)
                for filind, fil in enumerate(fillist):
                    dat = np.loadtxt(fil)
                    normdat = (dat[1]/dat[1][0])
                    alldat = alldat + normdat

                avgdat = alldat/len(fillist)
                if denseind == 4 and yp == 0:
                    axes[xp][yp].plot(dat[0], avgdat, color=colours[abundind], label='abund = ' + str(abund), alpha=1)
                    if abundind == 0:
                        axes[xp][yp].errorbar(obsdat[0], obsdat[1], obsdat[2], ls='', marker='.', label='Chandra data', color='black')

                axes[xp][yp].plot(dat[0], avgdat, color=colours[abundind])
                axes[xp][yp].errorbar(obsdat[0], obsdat[1], obsdat[2], ls='', marker='.', color='black')
                axes[xp][yp].set_title(f'Base density = {dense*5} $\\times 10^{{10}}$ atoms cm$^{{-3}}$')

            axes[xp][0].set_ylabel('Normalsied count rate'.format(dense*5, '{10}', '{-3}'))
            axes[-1][yp].set_xlabel('Orbital phase ($0.5$ is mid transit)')
        axes[-1][0].legend(loc='lower left')
        fig.suptitle(f'Scale height = {sclht} $R_\odot$')
        fig.tight_layout()
        fig.savefig(f'{path}sclht{sclht}/{savfil}')
        fig.show()
    return


def chigrids(sclhts, denses, abunds, path='chiarr_sclht_v3_single_aia.csv', savfil='chigrid_new.pdf'):
    # path changed from chiarr_sclht_v2.csv
    chiarr = pd.read_csv(path)
    chimin = np.min(chiarr['chi'].values)
    chimax = np.max(chiarr['chi'].values)
    print(chimin, chimax)
    nareas = chiarr.iloc[:0,:].copy()
    fig, axes = plt.subplots(3, 2, figsize=(8, 10), sharex=False, sharey=True)
    for aind, abund in enumerate(abunds):
        xp, yp = aind % 3, 0 if aind <=2 else 1
        print('here0')
        abundtab = chiarr[(abund == round(chiarr['abund'], 3))]
        print('here1')
        chim = np.zeros((len(denses), len(sclhts)))
        print('here2')
        rectarr = []
        for dind, dense in enumerate(denses):
            print('here3')
            densetab = abundtab[(dense*5e+10 == round(abundtab['dense'], 3))]
            for sind, sclht in enumerate(sclhts):
                print(sclht, dense*5e+10, abund)
                tabtouse = densetab[(sclht == round(densetab['sclht'], 3))]
                chi = tabtouse['chi'].values
                print(len(chi))
                if len(chi) > 0:
                    chim[dind][sind] = chi[0]
                    if chi[0] <= chimin * 1.05:
                        rectarr.append([sind, dind])
                        nareas = nareas.append(tabtouse, ignore_index=True)
                    if chi[0] == chimin:
                        print('here')
                        bestchi = [sind, dind]

        im = axes[xp][yp].imshow(chim, norm=LogNorm(vmin=chimin, vmax=chimax))
        for pos in rectarr:
            rect = patches.Rectangle((pos[0]-0.5, pos[1]-0.5), 1, 1, edgecolor='red', facecolor='none')
            axes[xp][yp].add_patch(rect)
        try:
            rect = patches.Rectangle((bestchi[0]-0.5, bestchi[1]-0.5), 1, 1, linewidth=2, edgecolor='gold', facecolor='none')
        except:
            fffffff =45
        axes[xp][yp].add_patch(rect)
        #plt.colorbar(im, ax=axes[xp][yp])
        axes[xp][yp].set_title(f'abund = {abund}')
        axes[-1][yp].set_xlabel(f'Scale height, $R_J$')
        axes[xp][yp].set_xticks(np.arange(len(sclhts)))
        axes[xp][yp].set_xticklabels(sclhts)
        axes[xp][0].set_ylabel(r'Density, $\times 5 \times 10^{10}$ atoms cm$^{-3}$')
        axes[xp][0].set_yticks(np.arange(len(denses)))
        axes[xp][0].set_yticklabels(denses)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    print(nareas)
    #fig.tight_layout()
    fig.suptitle('Identifying the best fitting zones of parameter space')
    fig.savefig(savfil)
    fig.show()
    #print(chiarr)


def ltc(nums, ht, den, ab):
    obsdat = getobsdat()
    plotdat = np.zeros(300)
    for i, num in enumerate(nums):
        data = np.loadtxt(f'data/sclht{ht}/dense{den}/abund{ab}/aianum{num}_band0.5-7.007_ltcdata.csv')
        plt.plot(data[0], data[1] / data[1][0], alpha=1, label=f'Aia Image {i}')
        plotdat += data[1]/data[1][0]
    plotdat = plotdat/60
    #plt.plot(data[0], plotdat, label='Average over all AIAs')
    #plt.errorbar(obsdat[0], obsdat[1], obsdat[2], marker='.', color='black', ls='', label='Observed')
    plt.legend(loc='lower left')
    plt.xlabel('Orbital phase')
    plt.ylabel('Normalised count rate')
    plt.title('Comparison between 2 different AIA file lightcurves')
    plt.tight_layout()
    plt.savefig('plots/compare.pdf')
    plt.show()


def plaex(abunds, denses, hts, path='sphrojcolfils 18 01 22'):

    outer = gridspec.GridSpec(1, len(abunds), wspace=0.3, hspace=0.0)
    fig = plt.figure(figsize=(18, 7))

    maxfil = glob.glob(f'{path}/sclht{hts[1]}/dense{denses[1]}/abund{abunds[1]}/*.csv')
    img = np.loadtxt(maxfil[0])
    maxd = np.max(img)*89*5e+10

    for ind, i in enumerate(outer):
        print(f'ind {ind}')
        if ind != 2:
            inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[ind])  # , wspace=0.1, hspace=0.1
            j = 0
            for denind, dense in enumerate(denses[:-1]):
                for htind, ht in enumerate(hts[:-1]):
                    print(f'dense {dense}')
                    fils = glob.glob(f'{path}/sclht{ht}/dense{dense}/abund{abunds[ind]}/*.csv')
                    img = np.loadtxt(fils[0])*dense*5e+10
                    ax = plt.Subplot(fig, inner[j])
                    ax, img = cropping(img, ax, False, maxd)
                    fig.add_subplot(ax)
                    if denind == 1:
                        ax.set_xlabel(('Coordinates, $R_\odot$  \nScale height = {} $R_\odot$\nAbundance = {} '.format(ht, abunds[ind])))
                    else:
                        ax.set_xticks([])
                    if htind == 0:
                        ax.set_ylabel('Base density = {} $\\times 10^{}$ atoms cm$^{}$ \nCoordinates, $R_\odot$'.format(dense*5, '{10}', '{-3}'))
                    else:
                        ax.set_yticks([])
                    j += 1
        else:
            inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[ind], wspace=0.1, hspace=0.1)
            fils = glob.glob(f'{path}/sclht{hts[-1]}/dense{denses[-1]}/abund{abunds[ind]}/*.csv')
            img = np.loadtxt(fils[0])*dense*5e+10
            ax = plt.Subplot(fig, inner[0])
            ax, img, im = cropping(img, ax, True, maxd)
            ax.set_xlabel(('Coordinates, $R_\odot$ \nScale height = {} $R_\odot$\nAbundance = {}\nBest fit case. '.format(hts[-1], abunds[ind])))
            ax.set_ylabel(
                'Base density = {} $\\times 10^{}$ atoms cm$^{}$ \nCoordinates, $R_\odot$'.format(denses[-1] * 5, '{10}',
                                                                                                 '{-3}'))
            plt.colorbar(im, ax=ax)
            fig.add_subplot(ax)
    fig.savefig('plots/ComparingAtmospheres.pdf')
    fig.savefig('plots/ComparingAtmospheres.png')
    fig.show()


def speccomp():
    specs = glob.glob('emissionspec_corkT*.csv')
    tot = np.zeros(446)
    fig, ax = plt.subplots(1,1)
    for fil in specs:
        print(fil)
        kt = fil.split('.')[1]
        spec = np.loadtxt(fil)

        tot += spec[1]
        plt.plot(spec[0], spec[1], label=f'kT = 0.{kt} keV')
    plt.plot(spec[0], tot, label=f'Summed spectrum')
    moddat = np.loadtxt('allout_model_data_cts.csv')
    plt.plot(moddat[0], moddat[1], ls='--', label='Fitted spectrum')
    plt.title('Out of transit spectrum')
    plt.legend(loc='lower left')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts/sec/keV')
    plt.ylim(1e-10, 1e-1)
    ax.set_yscale('log')

    plt.loglog()
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    ax.xaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.savefig('simfitspec.pdf')
    plt.show()


htlist = [0.01, 0.0301, 0.0502, 0.0702, 0.0903, 0.1103, 0.1304, 0.1505, 0.1705, 0.1906]
htlist2 = [0.1, 0.3, 0.5, 0.7]
denselist = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0]
abundlist = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
pbands = pd.read_csv('passbands2.csv')
e0 = pbands['e0'].values  # [0.5, 0.5, 0.8, 1.2, 2.3]#, 0.5, 0.3, 0.5, 0.2]
e1 = pbands['e1'].values  # [7.0, 0.8, 1.2, 2.3, 7]#, 1.5, 1.7, 1.7, 2.3]
lbands = ['full', 'soft', 'medium', 'hard', 'harder']#, 'r315', 'r515', 'r317', 'r517', 'SEEJ']
cbands = pbands['cbands'].values


#speccomp()
#starimg()
#lightcurves(False, './data/', [htlist[0], 0.0702, htlist[-1]], [denselist[0], denselist[-1]], abundlist, savfil='lightcurve_sclht_dense_abund_plot_b.pdf')#, bands=cbands)
print('going to chigrids')
dtab = chigrids(htlist2, denselist, abundlist)
##ltc([0, 25], 0.0702, 8.0, 3.0)
#'''for sclht in tqdm.tqdm(htlist[:-2]):
    #for dense in denselist:
        #for abund in abundlist:
            #planimg(abund, dense, sclht)'''

##plaex([1.0, 11.0, 11.0], [1.0, 89.0, 89.0], [0.01, 0.1906, 0.0502])

##modltc('./data/', htlist, denselist, abundlist)

#midtran = np.loadtxt('midtransitpanel.csv')
#fig, ax = plt.subplots(1,1, figsize=(40, 40))
#ax, cropped = cropping(midtran, ax, cb=False)
##plt.colorbar(im, ax=ax)
#ax.set_ylabel('Coordinates, $R_\odot$')
#ax.set_xlabel('Coordinates, $R_\odot$')
#ax.axis('off')
#fig.tight_layout()
#fig.savefig('cropmidpanel.pdf')
#fig.savefig('cropmidpanel.png')
#fig.show()
