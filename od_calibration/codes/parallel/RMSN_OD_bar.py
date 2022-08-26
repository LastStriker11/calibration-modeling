# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:00:51 2021

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
#=============================
R = [0.3,0.4,1]
reds = [0.7, 0.7, 0.7, 0.7]
rands = [0, 0.15, 0.3, 0.5]
rmsns = []
for rand,red in zip(rands,reds):
    suffix = '_'+str(int(red*10))+'_'+str(int(rand*100))
    true_od0 = pd.read_csv('spatial_temporal/true_od/curr_od'+suffix+'.csv', header=None)
    est_od0 = pd.read_pickle('spatial_temporal/random_reduction/counts_rmsn100_6'+suffix+'.pckl')[5]
    true_od0.columns = est_od0.columns
    true_od0 = true_od0.iloc[:,10:14]
    est_od0 = est_od0.iloc[:,10:14]
    rmsn_od0 = gof_eval(true_od0, est_od0)
    rmsns.append(rmsn_od0)
    del true_od0, est_od0, rmsn_od0

fig, ax = fig, ax = plt.subplots(figsize=(6,4))
rc('mathtext',fontset='cm')

ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

xs = list(range(4))

ax.bar([x-0.3 for x in xs], rmsns[0], width=0.2, edgecolor='k', color='white', label='$Red:0.7, Rand:0$')
ax.bar([x-0.1 for x in xs], rmsns[1], width=0.2, edgecolor='k', color='silver', label='$Red:0.7, Rand:0.15$')
ax.bar([x+0.1 for x in xs], rmsns[2], width=0.2, edgecolor='k', color='grey', label='$Red:0.7, Rand:0.3$')
ax.bar([x+0.3 for x in xs], rmsns[3], width=0.2, edgecolor='k', color='k', label='$Red:0.7, Rand:0.5$')

ax.set_xticks(xs)
ax.set_xticklabels(['7:00-7:15','7:15-7:30','7:30-7:45','7:45-8:00'])

ax.set_xlabel('Time interval')
ax.set_ylabel('Best OD RMSN')
ax.set_ylim([0.1,1.0])
ax.set_yticks(np.arange(0.1,1,0.2))
ax.legend(frameon=False, ncol=2)

fig.savefig('spatial_temporal/figures/RMSN_OD_bar_red_rand.pdf')
del rmsns
#%%
days = [10, 100, 200]
true_od = pd.read_csv('spatial_temporal/true_od/curr_od_7_15.csv', header=None)
true_od = true_od.iloc[:,10:14]
rmsns = []
for day in days:
    suffix = str(day)
    est_od0 = pd.read_pickle('spatial_temporal/history_dimension/counts_rmsn'+suffix+'_6.pckl')[5]
    est_od0 = est_od0.iloc[:,10:14]
    rmsn_od0 = gof_eval(true_od, est_od0)
    rmsns.append(rmsn_od0)
    del est_od0, rmsn_od0

fig, ax = fig, ax = universal_fig(figsize=(6,4))
rc('mathtext',fontset='cm')

ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

xs = list(range(4))

ax.bar([x-0.2 for x in xs], rmsns[0], width=0.2, edgecolor='k', color='white', label='$n_d:10$')
ax.bar([x for x in xs], rmsns[1], width=0.2, edgecolor='k', color='silver', label='$n_d:100$')
ax.bar([x+0.2 for x in xs], rmsns[2], width=0.2, edgecolor='k', color='grey', label='$n_d:200$')

ax.set_xticks(xs)
ax.set_xticklabels(['7:00-7:15','7:15-7:30','7:30-7:45','7:45-8:00'])

ax.set_xlabel('Time interval')
ax.set_ylabel('Best OD RMSN')
ax.set_ylim([0.1,1.0])
ax.set_yticks(np.arange(0.1,1,0.2))
ax.legend(loc='upper left')

fig.savefig('spatial_temporal/figures/RMSN_OD_bar_nd.pdf') 
del true_od, rmsns
#%%
reds = [0.3, 0.3, 0.3]
rands = [0.1, 0.3, 0.5]
# true_od = pd.read_csv('spatial_temporal/true_od/curr_od_7_15.csv', header=None)
true_od = pd.read_csv('spatial_temporal/true_od/curr_od_7_30.csv', header=None)
true_od = true_od.iloc[:,10:14]
rmsns = []
for red,rand in zip(reds,rands):
    suffix = str(red)+'_'+str(rand)
    # est_od0 = pd.read_pickle('spatial_temporal/Dm_variance/counts_rmsn100_6'+suffix+'.pckl')[5]
    est_od0 = pd.read_pickle('spatial_temporal/Dm_variance/0.7_0.3/counts_rmsn100_6'+suffix+'.pckl')[5]
    est_od0 = est_od0.iloc[:,10:14]
    rmsn_od0 = gof_eval(true_od, est_od0)
    rmsns.append(rmsn_od0)

fig, ax = fig, ax = universal_fig(figsize=(6,4))
rc('mathtext',fontset='cm')

ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

xs = list(range(4))

ax.bar([x-0.2 for x in xs], rmsns[0], width=0.2, edgecolor='k', color='white', label='$\sigma_{od},\sigma_t:0.1$')
ax.bar([x for x in xs], rmsns[1], width=0.2, edgecolor='k', color='silver', label='$\sigma_{od},\sigma_t:0.3$')
ax.bar([x+0.2 for x in xs], rmsns[2], width=0.2, edgecolor='k', color='grey', label='$\sigma_{od},\sigma_t:0.5$')

ax.set_xticks(xs)
ax.set_xticklabels(['7:00-7:15','7:15-7:30','7:30-7:45','7:45-8:00'])

ax.set_xlabel('Time interval')
ax.set_ylabel('Best OD RMSN')
ax.set_ylim([0.1,1.0])
ax.set_yticks(np.arange(0.1,1,0.2))
ax.legend(loc='upper left')

# fig.savefig('spatial_temporal/figures/RMSN_OD_bar_sigmas.pdf')
fig.savefig('spatial_temporal/figures/RMSN_OD_bar_sigmas_07_03.pdf')        
        
        
        
        
        
        
        
        
        
        
        
        
        