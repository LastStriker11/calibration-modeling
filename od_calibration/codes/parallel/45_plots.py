# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 08:33:18 2020

@author: lasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
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
def gof_mape(data0, data_simulated0):
    data0.columns = data_simulated0.columns
    mapes = []
    for i in range(data0.shape[1]):
        data = data0.loc[data0.iloc[:,i]>=3, data0.columns[i]]
        data_simulated = data_simulated0.loc[data.index, data0.columns[i]]
        sae = (np.absolute(data - data_simulated)/data).sum()
        n = data.shape[0]
        mape = sae/n
        mapes.append(mape)
    return mape
#%%
def col_name(data):
    t1 = list(np.arange(7,10,0.25))
    xs1 = [int(x) for x in t1]
    ys1 = [int(60*(x-int(x))) for x in t1]
    ys1 = ['00' if x==0 else str(x) for x in ys1]
    
    t2 = list(np.arange(7.25,10.25,0.25))
    xs2 = [int(x) for x in t2]
    ys2 = [int(60*(x-int(x))) for x in t2]
    ys2 = ['00' if x==0 else str(x) for x in ys2]
    
    cols = [str(x1)+':'+str(y1)+' - '+str(x2)+':'+str(y2)
            for x1,y1,x2,y2 in zip(xs1,ys1,xs2,ys2)]
    del t1, xs1, ys1, t2, xs2, ys2
    # draw the plot
    data.columns = cols
    return data
#%% 45 degree plots for OD's
def OD_45(true_od, est_od, best_rmsn, suffix):
    fig, ax = plt.subplots(figsize=(2.6,2.5))
    
    ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    ax.scatter(true_od, est_od, c='white', alpha=1, edgecolors='blue')
    ax.legend(['RMSN = %.2f'%best_rmsn], markerscale=0, frameon=False,
                handletextpad=0, handlelength=0, loc='lower right')
    ax.plot([0, true_od.max()+5], [0, true_od.max()+5], color = 'black', 
              linewidth = 1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Target OD matrix')
    ax.set_ylabel('Calibrated OD matrix')
    ax.text(0.5, 0.98, est_od.name, horizontalalignment='center', 
            verticalalignment='top', transform=ax.transAxes,
            bbox={'facecolor':'white', 'alpha':0.5, 'pad':1})
    ax.set_xlim([0,true_od.max()+5])
    ax.set_ylim([0,true_od.max()+5])
    fig.savefig('spatial_temporal/figures/OD_45/'+suffix[:2]+'_'+str(true_od.name)+'.pdf', bbox_inches='tight')
#=============================
R = [0.3,0.4,1]
for method in range(1,7):
    suffix = '_'+str(method)+str(R[0])+str(R[1])+str(R[2])
    true_od0 = pd.read_csv('spatial_temporal/curr_od_7_15.csv', header=None)
    est_od0 = pd.read_pickle('spatial_temporal/SPSA_paras_robust/a1c0.15/counts_rmsn100'+suffix+'.pckl')[5]
    true_od0.columns = est_od0.columns
    true_od0 = true_od0.iloc[:,10:]
    est_od0 = est_od0.iloc[:,10:]
    rmsn_od0 = gof_eval(true_od0, est_od0)
    est_od0 = col_name(est_od0)
    # rmsn_od0 = gof_mape(true_od0, est_od0)
    for j in range(len(rmsn_od0)):
        OD_45(true_od0.iloc[:,j], est_od0.iloc[:,j], rmsn_od0[j], suffix)
#%% 45 degree plots for counts
def counts_45(true_counts, est_counts, best_rmsn, suffix):
    fig, ax = plt.subplots(figsize=(2.5,2.5))
    rc('mathtext',fontset='cm')
    
    ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    ax.scatter(true_counts/1000, est_counts/1000, c='red', marker='x', alpha=1, edgecolors='black')
    ax.legend(['RMSN = %.2f'%best_rmsn], markerscale=0, frameon=False,
                handletextpad=0, handlelength=0, loc='lower right')
    num_max = round((true_counts.max()+5)/1000)
    if num_max%2 != 0:
        num_max = num_max - 1
    ax.plot([0, num_max], [0, num_max], color = 'black', 
              linewidth = 1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Target counts ($10^3$)')
    ax.set_ylabel('Calibrated counts ($10^3$)')
    ax.text(0.5, 0.98, est_counts.name, horizontalalignment='center', 
            verticalalignment='top', transform=ax.transAxes,
            bbox={'facecolor':'white', 'alpha':0.5, 'pad':1})
    ax.set_xlim([0,num_max])
    ax.set_ylim([0,num_max])
    ax.set_xticks([0, num_max/2, num_max])
    ax.set_yticks([0, num_max/2, num_max])
    fig.savefig('spatial_temporal/figures/counts_45/'+suffix[:2]+'_'+str(true_counts.name)+'.pdf', bbox_inches='tight')
#=============================
R = [0.3,0.4,1] 
for method in range(6,7):
    suffix = '_'+str(method)+str(R[0])+str(R[1])+str(R[2])
    true_counts0 = pd.read_pickle('spatial_temporal/SPSA_paras_robust/a1c0.15/counts_rmsn100'+suffix+'.pckl')[2]
    est_counts0 = pd.read_pickle('spatial_temporal/SPSA_paras_robust/a1c0.15/counts_rmsn100'+suffix+'.pckl')[3]
    best_rmsn = pd.read_pickle('spatial_temporal/SPSA_paras_robust/a1c0.15/counts_rmsn100'+suffix+'.pckl')[1]
    true_counts0 = true_counts0.iloc[:,8:]
    est_counts0 = est_counts0.iloc[:,8:]
    est_counts0 = col_name(est_counts0)
    for j in range(len(best_rmsn)-8):
        counts_45(true_counts0.iloc[:,j], est_counts0.iloc[:,j], best_rmsn[j+8], suffix)
#%% 45 degree plots for initial OD vs calibrated OD
def OD_45(init_od, est_od, suffix):
    fig, ax = plt.subplots(figsize=(2.6,2.5))
    
    ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    ax.scatter(init_od, est_od, c='g', alpha=1, edgecolors='g', marker='+')
    ax.plot([0, init_od.max()+5], [0, init_od.max()+5], color = 'black', 
             linewidth = 1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Initial OD matrix')
    ax.set_ylabel('Calibrated OD matrix')
    ax.text(0.5, 0.98, est_od.name, horizontalalignment='center', 
            verticalalignment='top', transform=ax.transAxes,
            bbox={'facecolor':'white', 'alpha':0.5, 'pad':1})
    ax.set_xlim([0,init_od.max()+5])
    ax.set_ylim([0,init_od.max()+5])
    fig.savefig('spatial_temporal/figures/OD_45_initial/init_OD'+suffix[:2]+'_'+str(init_od.name)+'.pdf', bbox_inches='tight')
#=============================
R = [0.3,0.4,1]
for method in range(6,7):
    suffix = '_'+str(method)+str(R[0])+str(R[1])+str(R[2])
    init_od0 = pd.read_pickle('spatial_temporal/SPSA_paras_robust/a1c0.15/counts_rmsn100'+suffix+'.pckl')[4]
    est_od0 = pd.read_pickle('spatial_temporal/SPSA_paras_robust/a1c0.15/counts_rmsn100'+suffix+'.pckl')[5]
    init_od0.columns = est_od0.columns
    init_od0 = init_od0.iloc[:,10:]
    est_od0 = est_od0.iloc[:,10:]
    est_od0 = col_name(est_od0)
    for j in range(init_od0.shape[1]):
        OD_45(init_od0.iloc[:,j], est_od0.iloc[:,j], suffix)
#%% 45 degree plots for initial OD vs target OD
def OD_45(init_od, target_od, suffix):
    fig, ax = plt.subplots(figsize=(2.6,2.5))
    
    ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    ax.scatter(init_od, target_od, c='m', alpha=1, edgecolors='m', marker='1')
    ax.plot([0, init_od.max()+5], [0, init_od.max()+5], color = 'black', 
             linewidth = 1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Initial OD matrix')
    ax.set_ylabel('Target OD matrix')
    ax.text(0.5, 0.98, target_od.name, horizontalalignment='center', 
            verticalalignment='top', transform=ax.transAxes,
            bbox={'facecolor':'white', 'alpha':0.5, 'pad':1})
    ax.set_xlim([0,init_od.max()+5])
    ax.set_ylim([0,init_od.max()+5])
    fig.savefig('spatial_temporal/figures/OD_45_initial_target/init_tag_OD'+suffix[:2]+'_'+str(init_od.name)+'.pdf', bbox_inches='tight')
#=============================
R = [0.3,0.4,1]
for method in range(6,7):
    suffix = '_'+str(method)+str(R[0])+str(R[1])+str(R[2])
    true_od0 = pd.read_csv('spatial_temporal/curr_od_7_15.csv', header=None)
    init_od0 = pd.read_pickle('spatial_temporal/SPSA_paras_robust/a1c0.15/counts_rmsn100'+suffix+'.pckl')[4]
    true_od0.columns = init_od0.columns
    true_od0 = true_od0.iloc[:,10:]
    init_od0 = init_od0.iloc[:,10:]
    true_od0 = col_name(true_od0)
    for j in range(init_od0.shape[1]):
        OD_45(init_od0.iloc[:,j], true_od0.iloc[:,j], suffix)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        