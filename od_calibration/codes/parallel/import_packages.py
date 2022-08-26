#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
from cycler import cycler
import matplotlib.ticker as mt
inline_rc = matplotlib.rcParams
import matplotlib.colors as colors
colors.colorConverter.cache = {}

from IPython.core.display import clear_output
import pickle
#from igraph import *
#import igraph.test
#print(igraph.__version__)
#igraph.test.run_tests()
#import cairo
import matplotlib.pyplot as plt
from pprint import pprint
#import psycopg2
#import numba

import numexpr
from collections import defaultdict
from sklearn.cluster import *
#from sklearn import metrics
from sklearn.decomposition import *

import pandas as pd
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import scipy.stats as ss
import scipy.io as sio
import statsmodels.api as sm
import sklearn as sl
pd.options.mode.chained_assignment = None  # default='warn'
matplotlib.rcParams.update(inline_rc)
from matplotlib.font_manager import FontProperties

import warnings
warnings.filterwarnings('ignore')

#print(matplotlib.rcParams)
import matplotlib.font_manager
[f for f in matplotlib.font_manager.fontManager.ttflist]
#print(matplotlib.rcParamsDefault)

# create colormaps
import seaborn as sns
sns.reset_orig()
get_ipython().run_line_magic('matplotlib', 'inline')
sns.palplot(sns.color_palette('deep',10))
sns.palplot(sns.color_palette("Set2",10))
sns.palplot(sns.color_palette("Paired",12))
[plblue,pblue,plgreen,pgreen,plred,pred,plorange,porange,plpurple,ppurple,plbrown,pbrown] = sns.color_palette("Paired",12)

import palettable
pal = palettable.colorbrewer.qualitative.Set1_9
colors = pal.mpl_colors
sns.palplot(colors)
[cred,cblue,cgreen,cpurple,corange,cyellow,cbrown,cpink,cgray] = colors
colors = cycler('color',colors)
matplotlib.rc('axes',prop_cycle=colors)
# plt.rcParams["axes.prop_cycle"] = cycler('color', 
#                     ['cred','cblue','cgreen','cpurple','corange','cyellow','cbrown','cpink','cgray'])

import matplotlib.colors as colors
colors.colorConverter.colors['cblue'] = cblue
colors.colorConverter.colors['cred'] = cred
colors.colorConverter.colors['cgreen'] = cgreen
colors.colorConverter.colors['corange'] = corange
colors.colorConverter.colors['cpink'] = cpink
colors.colorConverter.colors['cbrown'] = cbrown
colors.colorConverter.colors['cgray'] = cgray
colors.colorConverter.colors['cpurple'] = cpurple
colors.colorConverter.colors['cyellow'] = cyellow

colors.colorConverter.colors['plblue'] = plblue
colors.colorConverter.colors['pblue'] = pblue
colors.colorConverter.colors['plgreen'] = plgreen
colors.colorConverter.colors['pgreen'] = pgreen
colors.colorConverter.colors['plred'] = plred
colors.colorConverter.colors['pred'] = pred
colors.colorConverter.colors['plorange'] = plorange
colors.colorConverter.colors['porange'] = porange
colors.colorConverter.colors['plpurple'] = plpurple
colors.colorConverter.colors['ppurple'] = ppurple
colors.colorConverter.colors['plbrown'] = plbrown
colors.colorConverter.colors['pbrown'] = pbrown

colors.colorConverter.cache = {}

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import ipyparallel as ipp



def universal_fig(figsize=(3,3),fontsize=12,axislinewidth=1,markersize=5,text=None,limits=[-7,7],offset=[-44,12],projection=None, fontfamily=["Helvetica","Arial"], contain_latex=False):
    '''
    Create universal figure settings with publication quality
    returen fig, ax (similar to plt.plot)
    fig, ax = universal_fig()
    '''
    # ----------------------------------------------------------------
    if projection is None: fig,ax = plt.subplots(frameon = False)
    else: fig,ax = plt.subplots(frameon = False, subplot_kw=dict(projection=projection))
    fig.set_size_inches(figsize)
    matplotlib.rc("font",**{"family":"sans-serif", "sans-serif": fontfamily, "size": fontsize})
    matplotlib.rc('pdf', fonttype=42,use14corefonts=True,compression=6)
    matplotlib.rc('ps',useafm=True,usedistiller='none',fonttype=42)
    matplotlib.rc("axes",unicode_minus=False,linewidth=axislinewidth,labelsize='medium')
    matplotlib.rc("axes.formatter",limits=limits)
    matplotlib.rc('savefig',bbox='tight',format='eps',frameon=False,pad_inches=0.05)
    matplotlib.rc('legend')
    matplotlib.rc('lines',marker=None,markersize=markersize)
    matplotlib.rc('text',usetex=False)
    matplotlib.rc('xtick',direction='in')
    matplotlib.rc('xtick.major',size=4)
    matplotlib.rc('xtick.minor',size=2)
    matplotlib.rc('ytick',direction='in')
    matplotlib.rc('lines',linewidth=1)
    matplotlib.rc('ytick.major',size=4)
    matplotlib.rc('ytick.minor',size=2)
    matplotlib.rcParams['lines.solid_capstyle'] = 'butt'
    matplotlib.rcParams['lines.solid_joinstyle'] = 'bevel'
    matplotlib.rc('mathtext',fontset='stixsans')
    
    if contain_latex:
        matplotlib.rc('ps',useafm=False,usedistiller='none',fonttype=3)
        matplotlib.rc('pdf', fonttype=3,use14corefonts=True,compression=6)
        matplotlib.rc('text',usetex=True)
        
    
    matplotlib.rc('legend',fontsize='medium',frameon=False,
                  handleheight=0.5,handlelength=1,handletextpad=0.4,numpoints=1)
    if text is not None: 
        w = ax.annotate(text, xy=(0, 1), xycoords='axes fraction', fontsize='large',weight='bold',
                xytext=(offset[0]/12*fontsize, offset[1]/12*fontsize), textcoords='offset points', ha='left', va='top')
        print(w.get_fontname())
    # ----------------------------------------------------------------
    # end universal settings
    return fig, ax




def universal_fig_row_col(nrows=2,ncols=2,figsize=(3,3),sharex=True,sharey=True,fontsize=12,
    axislinewidth=1,markersize=5,text=None,limits=[-7,7],offset=[-44,12], fontfamily=["Helvetica","Arial"],contain_latex=False):
    '''
    Create universal figure settings with publication quality
    returen fig, ax (similar to plt.plot)
    fig, ax = universal_fig()
    '''
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(nrows, ncols,sharex=sharex, sharey=sharey)
    fig.set_size_inches(figsize)
    matplotlib.rc("font",**{"family":"sans-serif", "sans-serif": fontfamily, "size": fontsize})
    matplotlib.rc('pdf', fonttype=42,use14corefonts=True,compression=6)
    matplotlib.rc('ps',useafm=True,usedistiller='none',fonttype=42)
    matplotlib.rc("axes",unicode_minus=False,linewidth=axislinewidth,labelsize='medium')
    matplotlib.rc("axes.formatter",limits=limits)
    matplotlib.rc('savefig',bbox='tight',format='eps',frameon=False,pad_inches=0.05)
    matplotlib.rc('legend')
    matplotlib.rc('lines',marker=None,markersize=markersize)
    matplotlib.rc('text',usetex=False)
    matplotlib.rc('xtick',direction='in')
    matplotlib.rc('xtick.major',size=4)
    matplotlib.rc('xtick.minor',size=2)
    matplotlib.rc('ytick',direction='in')
    matplotlib.rc('lines',linewidth=1)
    matplotlib.rc('ytick.major',size=4)
    matplotlib.rc('ytick.minor',size=2)
    matplotlib.rcParams['lines.solid_capstyle'] = 'butt'
    matplotlib.rcParams['lines.solid_joinstyle'] = 'bevel'
    matplotlib.rc('mathtext',fontset='stixsans')
    
    if contain_latex:
        matplotlib.rc('ps',useafm=False,usedistiller='none',fonttype=3)
        matplotlib.rc('pdf', fonttype=3,use14corefonts=True,compression=6)
        
    
    
    matplotlib.rc('legend',fontsize='medium',frameon=False,
                  handleheight=0.5,handlelength=1,handletextpad=0.4,numpoints=1)
    if text is not None:
        if np.ndim(ax)>1:
            w = ax[0,0].annotate(text, xy=(0, 1), xycoords='axes fraction', fontsize='large',weight='bold',
                    xytext=(offset[0]/12*fontsize, offset[1]/12*fontsize), textcoords='offset points', ha='left', va='top')
            print(w.get_fontname())
        else:
            w = ax[0].annotate(text, xy=(0, 1), xycoords='axes fraction', fontsize='large',weight='bold',
                    xytext=(offset[0]/12*fontsize, offset[1]/12*fontsize), textcoords='offset points', ha='left', va='top')
            print(w.get_fontname())
    # ----------------------------------------------------------------
    # end universal settings
    return fig, ax



def log_bin_hist(data,min_v=1,st=0.1):
    '''data is a vector'''
    max_v = np.max(data)
    point = 1.5*10**st/(10**st-1)
    xx = np.arange(np.log10(point),np.log10(2*max_v),st)
    xx = np.floor(np.power(10,xx)+1)
    xx = np.concatenate((np.arange(min_v,np.min(xx)-1),xx),axis=0)
    y = np.histogram(data,xx)[0]
    x2 = np.diff(xx)
    xx = xx[:-1]
    x = xx + x2/2 * (x2>1)
    y = y/x2
    return y,x


def log_bin_average(x,y,min_v=1,st=0.1):
    '''data is a 2d-array (n*2) [x,y]'''
    x = x[x>0]
    y = y[x>0]
    max_v = np.max(x)
    point = 1.5*10**st/(10**st-1)
    xx = np.arange(np.log10(point),np.log10(2*max_v),st)
    xx = np.floor(np.power(10,xx)+1)
    xx = np.concatenate((np.arange(min_v,np.min(xx)-1),xx),axis=0)
    avg = np.zeros(len(xx)-1)
    x2 = np.diff(xx)
    for i in range(len(avg)):
        avg[i] = np.mean(y[(x>=xx[i]) & (x<xx[i+1])])
    xx = xx[:-1]
    x = xx + x2/2 * (x2>1)
    return avg,x


def jsd(xmat,ymat,eps=1e-9):
    '''
    Jensen-Shannon Divergence
    First resize x and y to prob vectors
    JSD = 0.5*(KL(x,z)+KL(y,z))
    in which z=(x+y)/2
    Note that jsd may accepted matrices (they will be flattened)
    '''
    x, y = np.asarray(xmat).ravel(), np.asarray(ymat).ravel()
    x = x/np.sum(x)
    y = y/np.sum(y)
    z = (x+y)/2
    return 0.5*(ss.entropy(x,z,base=2)+ss.entropy(y,z,base=2))


# In[ ]:




