# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:36:31 2020

@author: lasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from import_packages import *
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
#%% RMSN plots for interval vs. RMSN on OD (and counts) for different methods
def inter_method(rmsn_od, rmsn_counts):
    fig, ax1 = universal_fig(figsize=(6, 4))
    ax1.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    markers = ['D', 'v', 'x', 's', 'P', '*']
    colors = ['r','b','m','brown','orange','green']
    t1 = list(np.arange(6,10,0.5))
    xs1 = [int(x) for x in t1]
    ys1 = [int(60*(x-int(x))) for x in t1]
    ys1 = ['00' if x==0 else str(x) for x in ys1]
    inter = [str(x1)+':'+str(y1) for x1, y1 in zip(xs1,ys1)]
    for i in range(rmsn_od.shape[1]):
        ax1.plot(rmsn_od.iloc[:,i], c=colors[i], marker=markers[i],
                  fillstyle='none', linestyle='-', 
                  label='Method '+str(rmsn_od.columns[i]))
    ax1.set_xticks(list(range(0,18,2)))
    ax1.set_xticklabels(inter)
    ax1.set_xlabel('Intervals')
    ax1.set_ylabel('RMSN')
    ax1.set_ylim([0.1,1.1])
    ax1.legend(ncol=2)
    fig.savefig('spatial_temporal/figures/inter_method_OD.pdf')
    
    fig, ax2 = universal_fig(figsize=(6, 4))
    ax2.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['top'].set_linewidth(1.5)
    ax2.spines['right'].set_linewidth(1.5)
    for i in range(rmsn_counts.shape[1]):
        ax2.plot(rmsn_counts.iloc[:,i], c=colors[i], marker=markers[i],
                  fillstyle='none',linestyle='-',
                  label='Method '+str(rmsn_od.columns[i]))
    ax2.set_xticks(list(range(0,18,2)))
    ax2.set_xticklabels(inter)
    ax2.set_xlabel('Intervals')
    ax2.set_ylabel('RMSN')
    ax2.set_ylim([0.05,0.7])
    fig.savefig('spatial_temporal/figures/inter_method_counts.pdf')

#=============================
R = [0.3,0.4,1]
rmsn_od = pd.DataFrame()
rmsn_counts = pd.DataFrame()
for method in range(1,7):
    suffix = '_'+str(method)+str(R[0])+str(R[1])+str(R[2])
    true_od0 = pd.read_csv('spatial_temporal/curr_od.csv', header=None)
    est_od0 = pd.read_pickle('spatial_temporal/SPSA_paras_robust/a1c0.15/counts_rmsn100'+suffix+'.pckl')[5]
    rmsn_od0 = gof_eval(true_od0.iloc[:,6:], est_od0.iloc[:,6:])
    rmsn_od[str(method)] = rmsn_od0
    true_counts0 = pd.read_pickle('spatial_temporal/SPSA_paras_robust/a1c0.15/counts_rmsn100'+suffix+'.pckl')[2]
    est_counts0 = pd.read_pickle('spatial_temporal/SPSA_paras_robust/a1c0.15/counts_rmsn100'+suffix+'.pckl')[3]
    rmsn_counts0 = gof_eval(true_counts0.iloc[:,4:], est_counts0.iloc[:,4:])
    rmsn_counts[str(method)] = rmsn_counts0
inter_method(rmsn_od, rmsn_counts)
#%%
def col_name(data):
    t1 = list(np.arange(6,10,0.25))
    xs1 = [int(x) for x in t1]
    ys1 = [int(60*(x-int(x))) for x in t1]
    ys1 = ['00' if x==0 else str(x) for x in ys1]
    
    t2 = list(np.arange(6.25,10.25,0.25))
    xs2 = [int(x) for x in t2]
    ys2 = [int(60*(x-int(x))) for x in t2]
    ys2 = ['00' if x==0 else str(x) for x in ys2]
    
    cols = [str(x1)+':'+str(y1)+' - '+str(x2)+':'+str(y2)
            for x1,y1,x2,y2 in zip(xs1,ys1,xs2,ys2)]
    del t1, xs1, ys1, t2, xs2, ys2
    # draw the plot
    data.columns = cols
    return data
#%% comparsion for the same interval
def draw_interval(rmsn, inter, R):
    fig, ax = universal_fig(figsize=(6,4))
    
    ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    suffix = '_'+str(R[0])+str(R[1])+str(R[2])
    markers = ['D', 'v', 'x', 's', 'P', '*']
    colors = ['r','b','m','brown','orange','green']
    for i in range(6):
        ax.plot(rmsn.iloc[:,i], color=colors[i], fillstyle='none',
                marker=markers[i], linestyle='-')
    
    methods = list(range(1,7))
    # lengends = ['method'+str(m)+': '+rmsn.columns[0] for m in methods]
    lengends = ['method'+str(m) for m in methods]
    ax.legend(lengends)
    ax.text(0.5, 0.98, rmsn.columns[0], horizontalalignment='center', 
            verticalalignment='top', transform=ax.transAxes,
            bbox={'facecolor':'white', 'alpha':0.5, 'pad':1})
    
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('RMSN')
    # ax.set_ylim([0.1,1.0])
    # ax.set_yticks(np.arange(0.1,1,0.2))
    fig.savefig('spatial_temporal/figures/intervals/CC_15_Dm'+suffix+'_'+str(6+inter*0.25)+'.pdf')
#=============================
R = [0.3,0.4,1]
for inter in range(16):
    rmsn_inter = pd.DataFrame()
    for method in range(1,7):
        suffix = '_'+str(method)+str(R[0])+str(R[1])+str(R[2])
        rmsn0 = pd.read_pickle('spatial_temporal/SPSA_paras_robust/a1c0.15/counts_rmsn100'+suffix+'.pckl')[0]
        rmsn = rmsn0.iloc[:,4:]
        rmsn = col_name(rmsn)
        rmsn_inter = pd.concat([rmsn_inter, rmsn.iloc[:,inter]], axis=1)
    draw_interval(rmsn_inter, inter, R)
#%% comparsion of convergence within different intervals between different weights
def draw_interval_w(rmsn, inter):
    fig, ax = plt.subplots(figsize=(6,4))
    
    ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    markers = ['*','D', 'v', 'x', 's', 'P']
    colors = ['green','r','b','m','brown','orange']
    linestyles = ['-','--',':','-.']
    for i in range(4):
        ax.plot(rmsn.iloc[:,i], color=colors[i], fillstyle='none',
                marker=markers[i], linestyle=linestyles[i])
    
    weights = [0,0.2,0.4,0.6]
    # lengends = ['method'+str(m)+': '+rmsn.columns[0] for m in methods]
    lengends = ['$w_{od}$: '+str(w) for w in weights]
    ax.legend(lengends, loc='upper right')
    ax.text(0.5, 0.98, rmsn.columns[0], horizontalalignment='center', 
            verticalalignment='top', transform=ax.transAxes,
            bbox={'facecolor':'white', 'alpha':0.5, 'pad':1})
    
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('RMSN')
    # ax.set_ylim([0.1,1.0])
    # ax.set_yticks(np.arange(0.1,1,0.2))
    fig.savefig('spatial_temporal/figures/intervals/weights_'+str(6+inter*0.25)+'.pdf')

for inter in range(16):
    rmsn_inter = pd.DataFrame()
    for w in [0,0.2,0.4,0.6]:
        suffix = str(w)
        rmsn0 = pd.read_pickle('spatial_temporal/gof_eval_od/Munich_MR_od_'+suffix+'.pckl')[0]
        rmsn = rmsn0.iloc[:,4:]
        rmsn = col_name(rmsn)
        rmsn_inter = pd.concat([rmsn_inter, rmsn.iloc[:,inter]], axis=1)
    draw_interval_w(rmsn_inter, inter)






















