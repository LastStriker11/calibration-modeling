# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:28:30 2019

@author: User
"""

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
#%%
counts = pd.read_csv('t_test_counts100.csv', header=None)
time = pd.read_csv('t_test_time100.csv', header=None)
time.dropna(axis=0, inplace=True)
counts.dropna(axis=0, inplace=True)
#%%
counts['label'] = counts.iloc[:,0]
counts.drop(0, axis=1, inplace=True)
counts.set_index('label', inplace=True)

time['label'] = list(zip(time.iloc[:,0], time.iloc[:,1]))
time.drop([0,1], axis=1, inplace=True)
time.set_index('label', inplace=True)
#%% Only consider the links that matter
counts_mean = pd.DataFrame(counts.mean(axis=1))
index1 = counts_mean.iloc[:,0]>175
counts = counts.loc[index1,:]

time_mean = pd.DataFrame(time.mean(axis=1))
index2 = time_mean.iloc[:,0]>400
time = time.loc[index2,:]
#%%
alpha = 0.05
ci = 0.2
t_counts = pd.DataFrame()
t_time = pd.DataFrame()
for num in range(2, counts.shape[1]+1):
    t = stats.t.ppf(1-alpha/2, num-1)
    W_counts = counts.iloc[:,:num].mean(axis=1)*ci
    t_counts['std'] = counts.iloc[:,:num].std(axis=1)
    t_counts[str(num)] = np.ceil(((2*t*t_counts['std'])/W_counts)**2)
    
    W_time = time.iloc[:,:num].mean(axis=1)*ci
    t_time['std'] =time.iloc[:,:num].std(axis=1)
    t_time[str(num)] = np.ceil(((2*t*t_time['std'])/W_time)**2)
#%%
t_counts.to_csv('t_counts.csv')
t_time.to_csv('t_time.csv')
#%%
counts100 = t_counts.iloc[:,-1]
time100 = t_time.iloc[:,-1]

plt.rcParams.update({'figure.figsize':(8,8), 'figure.dpi':60, 'figure.autolayout': True})
plt.figure()
sns.distplot(counts100)
plt.title('Distribution of NSumoRep for counts')
plt.xlabel('No. of SUMO Simulation Replication')
plt.ylabel('Frequency')
plt.savefig('dist_counts.pdf')
plt.show()
plt.figure()
sns.distplot(time100)
plt.title('Distribution of NSumoRep for time')
plt.xlabel('No. of SUMO Simulation Replication')
plt.ylabel('Frequency')
plt.savefig('dist_time.pdf')
plt.show()