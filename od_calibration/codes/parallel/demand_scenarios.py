# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:06:52 2019
@author: lasts
"""
import pandas as pd
import numpy as np
from pathos.multiprocessing import ThreadingPool as TPool
import time
import os
from Functions import load_path, load_data, write_od, RSUMOStateInPar, SUMO_aver
import platform
#%%
if __name__ == '__main__':
    NSumoRep = 10
    network='your network'
    D_M_para = None
    scenario_type = 'spatial_temporal'
    system = platform.system()
    #%% generating scenario
    #==================================
    start = time.time()
    # load path and data
    Paths, Network_var, SPSA_para, PC_SPSA_para, Supply_para, Route_choice_para = load_path(network, system, True, 'counts', 'create', NSumoRep, 1, 1)
    data, input_od = load_data(Paths, Network_var)
    input_hour_od = input_od.copy()
    input_od = pd.DataFrame()
    for i in range(5):
        for j in range(4):
            input_od = pd.concat([input_od, input_hour_od.iloc[:,(i+2)]/4], axis=1)
    input_od.columns = list(np.arange(5,10,Network_var['interval']))
    input_od = round(input_od) 
    input_od = pd.concat([input_hour_od.iloc[:,:2], input_od], axis=1)
    print('======Scenario Generation=======')

    print('------Generating true measurements---')
    x = 1.0
    y = 0.15
    delta = 0.333
    od = input_od.iloc[:,2:]
    # spatial
    if scenario_type == 'spatial':
        curr_od = pd.DataFrame()
        for inter in range(od.shape[1]):
            r = np.random.normal(0, delta, size=(od.shape[0]))
            tmp_od = (x+y*r)*od.iloc[:,i]
            curr_od = pd.concat([curr_od,tmp_od], axis=1)
    # temporal
    if scenario_type == 'temporal':
        curr_od = pd.DataFrame()
        for pair in range(od.shape[0]):
            r = np.random.normal(0, delta, size=(od.shape[1]))
            tmp_od = (x+y*r)*od.iloc[pair,:]
            curr_od = pd.concat([curr_od,pd.DataFrame(tmp_od).T], axis=0)
    # spatial-temporal
    if scenario_type == 'spatial_temporal':
        r = np.random.normal(0, delta, size=(od.shape[0],od.shape[1]))
        curr_od = (x+y*r)*od

    curr_od = pd.concat([input_od.iloc[:,:2], curr_od.astype('int32')], axis=1)
    tmap = TPool().map
    PATH, NETWORK, COUNTER, seedNN_vector, GENERATION = write_od(Paths, Network_var, curr_od, 'create')
    tmap(RSUMOStateInPar, PATH, NETWORK, [Supply_para]*10, [Route_choice_para]*10, COUNTER, seedNN_vector, GENERATION)
    simulated_counts, simulated_time = SUMO_aver(Paths, Network_var, 'create')
    if not os.path.exists(Paths['origin_network'] + 'scenarios/'):
        os.mkdir(Paths['origin_network'] + 'scenarios')
    suffix = 'scenarios/'+scenario_type+'/'+str(x)+'_'+str(y)
    if not os.path.exists(Paths['origin_network'] + suffix):
        os.mkdir(Paths['origin_network'] + suffix)
        os.mkdir(Paths['origin_network'] +suffix+ '/counts')
        os.mkdir(Paths['origin_network'] +suffix+ '/time')
    for i in range(simulated_counts.shape[1]):
        counts = simulated_counts.loc[:, simulated_counts.columns[i]]
        travel_time = simulated_time.loc[:, simulated_time.columns[i]]
        if not i == (simulated_counts.shape[1]-1):
            counts = counts + simulated_counts.loc[:, simulated_counts.columns[i+1]]
            travel_time = travel_time + simulated_time.loc[:, simulated_time.columns[i+1]]
        counts.to_csv(Paths["origin_network"] + suffix+ '/counts/'+ str(input_od.columns[i+2])+'true_counts.csv', header=False)
        travel_time.to_csv(Paths["origin_network"] + suffix+ '/time/'+ str(input_od.columns[i+2])+'true_tt.csv', header=False)            
    curr_od.to_csv(Paths["origin_network"] + suffix +'/curr_od.csv', index=False, header=False)
    print('======Done!===========')
    end = time.time()
    print('Time for Generating the scenarios: %d s.' %(end-start))