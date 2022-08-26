# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:20:32 2019

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

foler = '../parallel/results'
files = [f for f in os.listdir(foler) if f.endswith('.pckl')]

if not os.path.exists('figures'):
    os.mkdir('figures')

#%% heatmap for x
for h in range(5):
    rmsn = pd.DataFrame()
    yticks = []
    for f in [f for f in files if '_2_4' in f]:
        temp_rmsn = pd.read_pickle(foler+'/'+f)[0]
        rmsn = pd.concat([rmsn, temp_rmsn.iloc[:,h]], axis=1)
        yticks.append(int(f.split('_')[2])/10)
    plt.subplot(3,2,(h+1))
    sns.heatmap(rmsn.T, vmin=0, vmax=1.5, cmap="YlGnBu",\
                yticklabels=yticks, cbar=False, annot=True)
    plt.xlabel('No. of Iterations')
    plt.ylabel('Value of x')
plt.subplots_adjust(wspace=0.3, hspace=0.8)
plt.savefig('figures/D_M_x.pdf')
plt.show()
#%% heatmap for y
for h in range(5):
    rmsn = pd.DataFrame()
    yticks = []
    for f in [f for f in files if ('8_' in f)&('_6' in f)]:
        temp_rmsn = pd.read_pickle(foler+'/'+f)[0]
        rmsn = pd.concat([rmsn, temp_rmsn.iloc[:,h]], axis=1)
        yticks.append(int(f.split('_')[3])/10)
    plt.subplot(3,2,(h+1))
    sns.heatmap(rmsn.T, vmin=0, vmax=1.5, cmap="YlGnBu",\
                yticklabels=yticks, cbar=False, annot=True)
    plt.xlabel('No. of Iterations')
    plt.ylabel('Value of y')
plt.subplots_adjust(wspace=0.3, hspace=0.8)
plt.savefig('figures/D_M_y.pdf')
plt.show()
#%% heatmap for sigma
for h in range(5):
    rmsn = pd.DataFrame()
    yticks = []
    for f in [f for f in files if '8_2_' in f]:
        temp_rmsn = pd.read_pickle(foler+'/'+f)[0]
        rmsn = pd.concat([rmsn, temp_rmsn.iloc[:,h]], axis=1)
        yticks.append(int(f.split('_')[4])/10)
    plt.subplot(3,2,(h+1))
    sns.heatmap(rmsn.T, vmin=0, vmax=1.5, cmap="YlGnBu",\
                yticklabels=yticks, cbar=False, annot=True)
    plt.xlabel('No. of Iterations')
    plt.ylabel('Value of sigma')
plt.subplots_adjust(wspace=0.3, hspace=0.8)
plt.savefig('figures/D_M_sigma.pdf')
plt.show()
   
