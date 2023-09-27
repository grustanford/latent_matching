import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

ORIGIN = str(Path(os.path.abspath(__file__)).parent.parent.parent.absolute())
DIR_FIGURE = str(Path(ORIGIN+'/assets'))

def mns(vec, ci=True):
    _m = np.nanmean(vec,axis=-1)
    _n = np.sum(~np.isnan(vec),axis=-1)
    _s = np.nanstd(vec,axis=-1)/np.sqrt(_n)
    if ci:
        _s *= 1.96
    return _m, _s

def draw_publish_axis(ax, xrange, yrange, xticks, yticks, xwidth=2.5, ywidth=2):
    _xmin, _ = ax.get_xaxis().get_view_interval()
    _ymin, _ = ax.get_yaxis().get_view_interval()
    if xrange is not None:
        ax.add_artist(mlines.Line2D(xrange, (_ymin,_ymin), color='black', linewidth=xwidth, solid_capstyle='butt', fillstyle='full'))
    if yrange is not None:
        ax.add_artist(mlines.Line2D((_xmin,_xmin), yrange, color='black', linewidth=ywidth, solid_capstyle='butt', fillstyle='full'))
    ax.set_frame_on(False)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    
def setup_matplotlib(setup_type='article'):
    """setup matplotlib parameters for publishable quality figures
    https://github.com/adrian-valente/populations_paper_code
    """
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.rcParams['axes.titlepad'] = 12
    plt.rcParams['axes.labelpad'] = 5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['font.family'] = 'Helvetica'

    if setup_type=='article':
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['font.size'] = 12
    elif setup_type=='poster':
        plt.rcParams['axes.labelsize'] = 25
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['font.size'] = 18
