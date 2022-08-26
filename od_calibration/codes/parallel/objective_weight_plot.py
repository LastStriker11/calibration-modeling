# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:34:54 2021

@author: Qing-Long Lu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
#%%
def gof_eval(data0, data_simulated0):
    data0.columns = data_simulated0.columns
    rmsns = []
    for i in range(data0.shape[1]):
        data = data0.loc[data0.iloc[:,0]>=3, data0.columns[i]]
        data_simulated = data_simulated0.loc[data.index, data0.columns[i]]
        diff = (data - data_simulated)**2
        n = diff.count()
        sum_diff = diff.sum()
        sum_true = data.sum()
        rmsn = np.sqrt(n*sum_diff)/sum_true
        rmsns.append(rmsn)
    return rmsns
#%%
def inter_method(rmsn_od, rmsn_counts):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    markers = ['*', 'D', 'v', 'x', 's', 'P']
    colors = ['g','r','b','m','brown','orange']
    linestyles = ['-','--',':','-.']
    t1 = list(np.arange(6,10,0.5))
    xs1 = [int(x) for x in t1]
    ys1 = [int(60*(x-int(x))) for x in t1]
    ys1 = ['00' if x==0 else str(x) for x in ys1]
    inter = [str(x1)+':'+str(y1) for x1, y1 in zip(xs1,ys1)]
    for i in range(rmsn_od.shape[1]):
        ax1.plot(rmsn_od.iloc[:,i], c=colors[i], marker=markers[i],
                  fillstyle='none', linestyle=linestyles[i], 
                  label='$w_{od}$: '+str(rmsn_od.columns[i]))
    ax1.set_xticks(list(range(0,16,2)))
    ax1.set_xticklabels(inter)
    ax1.set_xlabel('Intervals')
    ax1.set_ylabel('RMSN')
    ax1.set_ylim([0.1,1.1])
    ax1.legend(ncol=2)
    fig.savefig('spatial_temporal/figures/weights_method_OD.pdf')
    
    fig, ax2 = plt.subplots(figsize=(6, 4))
    ax2.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['top'].set_linewidth(1.5)
    ax2.spines['right'].set_linewidth(1.5)
    for i in range(rmsn_counts.shape[1]):
        ax2.plot(rmsn_counts.iloc[:,i], c=colors[i], marker=markers[i],
                  fillstyle='none', linestyle=linestyles[i], 
                  label='$w_{od}$: '+str(rmsn_od.columns[i]))
    ax2.set_xticks(list(range(0,16,2)))
    ax2.set_xticklabels(inter)
    ax2.set_xlabel('Intervals')
    ax2.set_ylabel('RMSN')
    ax2.set_ylim([0.05,1.35])
    fig.savefig('spatial_temporal/figures/weights_method_counts.pdf')
#%%
#=============================
rmsn_od = pd.DataFrame()
rmsn_counts = pd.DataFrame()
for w in [0,0.2,0.4,0.6]:
    suffix = str(w)
    true_od0 = pd.read_csv('spatial_temporal/curr_od_7_15.csv', header=None)
    est_od0 = pd.read_pickle('spatial_temporal/gof_eval_od/Munich_MR_od_'+suffix+'.pckl')[5]
    rmsn_od0 = gof_eval(true_od0.iloc[:,6:], est_od0.iloc[:,6:])
    rmsn_od[str(w)] = rmsn_od0
    true_counts0 = pd.read_pickle('spatial_temporal/gof_eval_od/Munich_MR_od_'+suffix+'.pckl')[2]
    est_counts0 = pd.read_pickle('spatial_temporal/gof_eval_od/Munich_MR_od_'+suffix+'.pckl')[3]
    rmsn_counts0 = gof_eval(true_counts0.iloc[:,4:], est_counts0.iloc[:,4:])
    rmsn_counts[str(w)] = rmsn_counts0
inter_method(rmsn_od, rmsn_counts)