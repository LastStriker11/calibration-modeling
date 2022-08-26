# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 19:45:08 2021

@author: lasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from import_packages import *
#%%
data_od = pd.read_csv('spatial_temporal/curr_od_7_15.csv', header=None)

data_demand = data_od.iloc[:,6:].sum()
data_demand = data_demand.values.reshape(-1,4).sum(1)
#%%
fig, ax = universal_fig(figsize=(4,3))
rc('mathtext',fontset='cm')
    
ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

# bars = [str(t) for t in np.arange(6,10,0.25)]
ax.bar(range(4), data_demand/1000, color='white', edgecolor='k', width=0.7)
ax.set_xlabel('Time')
ax.set_ylabel('Demand ($10^3$)')
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(['6','7','8','9'])
fig.savefig('spatial_temporal/demand_7_15.pdf')