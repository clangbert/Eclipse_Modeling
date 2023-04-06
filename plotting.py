
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.ticker
import matplotlib.patches as patches

def chigrids(sclhts, densegrid, abundgrid, path='chiarr_sclht_v3._single_aia.csv', savfil='plots/chigrid_new.pdf'):
    chiarr = pd.read_csv(path)
    chimin = np.min(chiarr['chi'].values)
    chimax = np.max(chiarr['chi'].values)
    print(chimin, chimax)
    dorect = True
    nareas = chiarr.iloc[:0,:].copy()
    fig, axes = plt.subplots(3, 2, figsize=(8, 10), sharex=False, sharey=False)
    for sind, sclht in enumerate(sclhts):  #aind, abund in enumerate(abunds):
        xp, yp = sind % 3, 0 if sind <=2 else 1
        print('here0')
        sclhttab = chiarr[(sclht == round(chiarr['sclht'], 3))]
        print('here1')
        denses = densegrid[sind]
        abunds = abundgrid[sind]
        chim = np.zeros((len(denses), len(abunds)))
        print('here2')
        rectarr = []
        #print('sclht',sclhttab)
        for dind, dense in enumerate(denses):
            print('here3')
            densetab = sclhttab[(dense*1e+10 == round(sclhttab['dense'], 3))]
            #print('dense',densetab)
            for aind, abund in enumerate(abunds):
                print(sclht, dense*1e+10, abund)
                tabtouse = densetab[(abund == round(densetab['abund'], 3))]
                chi = tabtouse['chi'].values
                print('abund',tabtouse, chi)
                if len(chi) > 0:
                    chim[dind][aind] = chi[0]
                    print(chi/chimin, chimin)
                    if chi[0] <= chimin * 1.05:
                        rectarr.append([aind, dind])
                        nareas = nareas.append(tabtouse, ignore_index=True)
                    if chi[0] == chimin:
                        print('here')
                        bestchi = [aind, dind]
        print(xp, yp)
        im = axes[xp][yp].imshow(chim, norm=LogNorm(vmin=chimin, vmax=chimax))
        print(rectarr)
        divider = make_axes_locatable(axes[xp][yp])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')#, label='Fit stat (chi)')
        for pos in rectarr:
            rect = patches.Rectangle((pos[0]-0.5, pos[1]-0.5), 1, 1, edgecolor='red', facecolor='none')
            print('rect defined', rectarr)
            axes[xp][yp].add_patch(rect)
        try:
            if dorect:
                rect = patches.Rectangle((bestchi[0]-0.5, bestchi[1]-0.5), 1, 1, linewidth=2, edgecolor='gold', facecolor='none')
                axes[xp][yp].add_patch(rect)
                dorect=False
        except:
            fffffff =45

        #plt.colorbar(im, ax=axes[xp][yp])
        axes[xp][yp].set_title(f'Scale height = {sclht} $R_J$')
        axes[-1][yp].set_xlabel(f'Abundance')
        axes[xp][yp].set_xticks(np.arange(len(abunds)))
        axes[xp][yp].set_xticklabels(abunds)
        axes[xp][0].set_ylabel(r'Density, $/ 10^{10}$ atoms cm$^{-3}$')
        axes[xp][yp].set_yticks(np.arange(len(denses)))
        axes[xp][yp].set_yticklabels(denses)
    axes[2][1].axis('off')
    axes[1][1].set_xlabel(f'Abundance')
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #cb = fig.colorbar(im, cax=cbar_ax)
    print(nareas)
    fig.tight_layout()
    #fig.suptitle('Identifying the best fitting zones of parameter space')
    fig.savefig(savfil)
    fig.show()
    #print(chiarr)

hts = [0.1,
       0.3,
       0.5,
       0.7,
       0.9]
densearr = np.array([[5, 10, 15, 25, 40, 65, 105, 170, 275, 445],
                   [5, 10, 15, 25, 40, 65, 105, 170, 275, 445],
                   [5, 10, 15, 25, 40, 65, 105, 170, 275, 445],
                   [5, 10, 15, 25, 40, 65, 105, 170, 275, 445],
                   [5, 10, 15, 25, 40, 65, 105, 170, 275, 445]])
abarray = np.array([[1, 3, 5, 7, 9, 11],
                    [1, 3, 5, 7, 9, 11],
                    [1, 3, 5, 7, 9, 11],
                    [1, 3, 5, 7, 9, 11],
                    [1, 3, 5, 7, 9, 11]])

chigrids(hts, densearr, abarray, path='chiarr_sclht_v3.csv', savfil='plots/chigrid_old.png')
