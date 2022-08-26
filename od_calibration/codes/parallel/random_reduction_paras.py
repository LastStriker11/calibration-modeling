# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 11:27:23 2021

@author: lasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from import_packages import *
#%%
def col_name(data):
    t1 = list(np.arange(5,10,0.25))
    xs1 = [int(x) for x in t1]
    ys1 = [int(60*(x-int(x))) for x in t1]
    ys1 = ['00' if x==0 else str(x) for x in ys1]
    
    t2 = list(np.arange(5.25,10.25,0.25))
    xs2 = [int(x) for x in t2]
    ys2 = [int(60*(x-int(x))) for x in t2]
    ys2 = ['00' if x==0 else str(x) for x in ys2]
    
    cols = [str(x1)+':'+str(y1)+' AM - '+str(x2)+':'+str(y2)+' AM'
            for x1,y1,x2,y2 in zip(xs1,ys1,xs2,ys2)]
    del t1, xs1, ys1, t2, xs2, ys2
    # draw the plot
    data.columns = cols
    return data
#%%
def draw_method(rmsn, lenMark, method):
    fig, ax = universal_fig(figsize=(6,4))
    rc('mathtext',fontset='cm')
    
    ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    markers = ['D', 'v', 'x', 's']
    for i in range(4):
        ax.plot(rmsn.iloc[:,i], color='red', fillstyle='none', marker=markers[i])
        
    for i in range(4,8):
        ax.plot(rmsn.iloc[:,i], color='blue', fillstyle='none', marker=markers[i-4], linestyle='--')
        
    for i in range(8,12):
        ax.plot(rmsn.iloc[:,i], color='green', fillstyle='none', marker=markers[i-8], linestyle=':')
    
    # for i in range(12,16):
    #     ax.plot(rmsn.iloc[:,i], color='m', fillstyle='none', marker=markers[i-12], linestyle=(0, (3, 5, 1, 5)))
    
    custom_lines = [Line2D([0], [0], color='red', lw=1),
                Line2D([0], [0], color='blue', lw=1, linestyle='--'),
                Line2D([0], [0], color='green', lw=1, linestyle=':'),
                Line2D([0], [0], color='m', lw=1, linestyle=(0, (3, 5, 1, 5))),
                Line2D([0], [0], color='black', fillstyle='none',  marker='D'),
                Line2D([0], [0], color='black', fillstyle='none',  marker='v'),
                Line2D([0], [0], color='black', fillstyle='none',  marker='x'),
                Line2D([0], [0], color='black', fillstyle='none',  marker='s')]
    ax.legend(custom_lines, lenMark)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('RMSN')
    ax.set_ylim([0.1,1.0])
    ax.set_yticks(np.arange(0.1,1,0.2))
    fig.savefig('spatial_temporal/figures/red_diff_rand_15_method'+str(method)+'.pdf')
    # fig.savefig('spatial_temporal/figures/red_7_rand_diff_method'+str(method)+'.pdf')
#=============================  
reds = [0.7, 0.9, 1.2]
rands = [0.15, 0.15, 0.15]
# reds = [0.7, 0.7, 0.7, 0.7]
# rands = [0, 0.15, 0.3, 0.5]
method = 6
rmsn = pd.DataFrame()
lenMark = []
for rand,red in zip(rands,reds):
    suffix = '_'+str(int(red*10))+'_'+str(int(rand*100))
    lenMark.append('$Red$: '+str(red)+', $Rand$: '+str(rand))
    rmsn0 = pd.read_pickle('spatial_temporal/random_reduction/counts_rmsn100_'+str(method)+suffix+'.pckl')[0]
    rmsn0 = col_name(rmsn0)
    rmsn = pd.concat ([rmsn, rmsn0.iloc[:,8:12]], axis=1)
rmsn.dropna(axis=0, inplace=True)
lenMark = lenMark + rmsn.columns.tolist()[-4:]
draw_method(rmsn, lenMark, method)



























