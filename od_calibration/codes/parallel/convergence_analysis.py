# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:24:12 2020

@author: lasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from import_packages import *
#%%
def col_name(data):
    t1 = list(np.arange(6,8,0.25))
    xs1 = [int(x) for x in t1]
    ys1 = [int(60*(x-int(x))) for x in t1]
    ys1 = ['00' if x==0 else str(x) for x in ys1]
    
    t2 = list(np.arange(6.25,8.25,0.25))
    xs2 = [int(x) for x in t2]
    ys2 = [int(60*(x-int(x))) for x in t2]
    ys2 = ['00' if x==0 else str(x) for x in ys2]
    
    cols = [str(x1)+':'+str(y1)+' - '+str(x2)+':'+str(y2)
            for x1,y1,x2,y2 in zip(xs1,ys1,xs2,ys2)]
    del t1, xs1, ys1, t2, xs2, ys2
    # draw the plot
    data.columns = cols
    return data
#%%
def draw_method(rmsn, suffix):
    fig, ax = universal_fig(figsize=(6,4))
    
    ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    markers = ['D', 'v', 'x', 's']
    for i in range(4):
        ax.plot(rmsn.iloc[:,i], color='red', fillstyle='none',marker=markers[i])
    for i in range(4,8):
        ax.plot(rmsn.iloc[:,i], color='blue', fillstyle='none',marker=markers[i-4], linestyle='--')
    
    ax.legend(rmsn.columns, edgecolor='k', handlelength=1, ncol=2)
    ax.set_yticks(np.arange(0,1.4,0.2))
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('RMSN')
    plt.savefig('spatial_temporal/figures/methods'+suffix+'_6-7.pdf')
#=============================  
R = [0.3,0.4,1]
for method in range(1,7):
    suffix = '_'+str(method)+str(R[0])+str(R[1])+str(R[2])
    rmsn0 = pd.read_pickle('spatial_temporal/SPSA_paras_robust/a1c0.15/counts_rmsn100'+suffix+'.pckl')[0]
    rmsn = rmsn0.iloc[:,4:12]
    rmsn = col_name(rmsn)
    draw_method(rmsn, suffix)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    