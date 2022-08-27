# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:06:52 2019

@author: lasts
"""
import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessingPool as PPool
from pathos.multiprocessing import ThreadingPool as TPool
import pickle
import matplotlib.pyplot as plt
import time
import os
from Functions import load_path, load_data, write_od, RSUMOStateInPar, SUMO_aver, gof_eval_od, create_Dm
from Functions import generation_PC_SPSA as generation
from Functions import rep_generation_PC_SPSA as rep_generation
#%%
if __name__ == '__main__':
    network = 'Munich_MR'
    OD_reduction = 0.7
    method = 5 # 1 is Rand_ij, 2 is Rand_ij and Rand_d, 3 is Rand_t, 4 is Rand_d * Rand_t, 5 is Rand_matrix
    R = [0.3,0.4,1] # distribution factors Spatial, temporal, days
    #%% generating scenario
    Paths, Network_var, SPSA_para, PC_SPSA_para = load_path(
        network, False, 'counts', 'create', 10, 1, 1)
    data, input_od = load_data(Paths, Network_var)
    input_hour_od = input_od.copy()
    input_od = pd.DataFrame()
    for i in range(5):
        for j in range(4):
            input_od = pd.concat([input_od, input_hour_od.iloc[:,(i+2)]/4], axis=1)
    input_od.columns = list(np.arange(5,10,Network_var['interval']))
    input_od = round(input_od)
    input_od.iloc[:,2:] = round(input_od.iloc[:,2:]*OD_reduction) # OD is reduced to 70%
    input_od = pd.concat([input_hour_od.iloc[:,:2], input_od], axis=1)
    print('------Generating Dm--------')
    hist_od = create_Dm(Paths, Network_var, input_od, method, R)
    print('------Dm has been created----')
    #%% decomposing the historical od matrix
    start = time.time()
    Paths, Network_var, SPSA_para, PC_SPSA_para = load_path(
        network, False, 'counts', method, 10, 1, 15, 0.7, 0.5)
    data, input_od = load_data(Paths, Network_var)
    
    convergence = PC_SPSA_para["Min_error"]
    a = PC_SPSA_para["a"]       
    c = PC_SPSA_para["c"]
    A = PC_SPSA_para["A"]
    alpha = PC_SPSA_para["alpha"]
    gamma = PC_SPSA_para["gamma"]
    G = PC_SPSA_para["G"]
    N = PC_SPSA_para["N"]
    V = []
    z = []
    n_compon = []
    temp_U, temp_S, temp_V = np.linalg.svd(hist_od, full_matrices=False)
    temp_cv = temp_S.cumsum()/temp_S.sum()
    for temp_compon, score in enumerate(temp_cv): # find n_compon which can lead to a score > 0.95
        if score > 0.95:
            break
    temp_V = temp_V[:temp_compon,:]
    for i in range(len(Network_var["inputOD"])):
        temp_z = np.matmul(temp_V, input_od.iloc[:,i+2].values)
        V.append(temp_V.T)
        z.append(temp_z)
        n_compon.append(temp_compon)
    
    #%% initialization
    start_sim = int(float(Network_var["starttime"]))
    Best_OD = input_od.copy()
    Best_RMSN = [100]*(input_od.shape[1]-2)
    Best_SimulatedCounts = data.copy()
    rmsn = [] # to store the rmsn of each iteration
    index_same = input_od.iloc[:,2:]<5
    #%% the starting simulation
        
    print('Simulation 0 started')
    n_sumo = Network_var['NSumoRep']
    start_one = time.time()
    
    pamap = PPool(G).amap # asynchronous processing
    tmap = TPool(n_sumo).map # synachronous processing
    PATH, NETWORK, COUNTER, seedNN_vector, GENERATION = write_od(Paths, Network_var, input_od, 'start')
    
    tmap(RSUMOStateInPar, PATH, NETWORK, COUNTER, seedNN_vector, GENERATION)
    #=============================='''
    data_simulated = SUMO_aver(Paths, Network_var, 'start')
#    data_simulated.dropna(axis=0, inplace=True)
    print('Simultaion 0 completed')
    y = gof_eval_od(data, data_simulated)
    rmsn.append(y)
    end_one = time.time()
    print('Starting RMSN = ', y)
    print('========================================')
    #%%
    for iteration in range(1, N + 1):
        # calculating gain sequence parameters
        ak = a / ((iteration + A) ** alpha)
        ck = c / (iteration ** gamma)
        
        GA, INPUT_OD, CK, DATA, N_COMPON, ZS, VS, INDEX_SAME, PATH, NETWORK \
        = rep_generation(G, input_od, ck, data, n_compon, z, V, index_same, Paths, Network_var)
        # the 'outer' parallel processing
        G_hat = np.stack(pamap(generation, GA, INPUT_OD, CK, DATA, N_COMPON, ZS, VS, \
                                  INDEX_SAME, PATH, NETWORK, [tmap]*G).get(), axis=-1)
        
        g_hat_it = np.zeros((max(n_compon), len(Network_var["inputOD"])))
        for i in range(G):
            temp_g = pd.DataFrame(G_hat[:,:,i])
            g_hat_it = g_hat_it + temp_g
        g_hat_it = pd.DataFrame(g_hat_it/G).T
        
        # minimization
        OD_min = pd.DataFrame()
        for i in range(len(Network_var["inputOD"])):
            g_hat = g_hat_it.iloc[i,:].dropna()
            z_per = z[i] - np.multiply(z[i],(ak*g_hat.values))
            temp_per = pd.DataFrame(np.matmul(V[i],z_per)) # temp per is the OD minimzied
            temp_per = pd.concat([input_od.iloc[:,:2], temp_per], axis=1)
            temp_per.columns = ['from', 'to', 'counts']
            temp_per.loc[index_same.iloc[:,i], 'counts'] = input_od.loc[index_same.iloc[:,i], start_sim+i*Network_var['interval']] # index_same is the index of ODs less than 5
            index_neg = temp_per.loc[:,'counts']<3
            temp_per.loc[index_neg, 'counts'] = input_od.loc[index_neg, start_sim+i*Network_var['interval']]
            OD_min = pd.concat([OD_min, temp_per['counts']], axis=1)
            z[i] = z_per.copy()
            
        OD_min = pd.concat([input_od.iloc[:,:2], OD_min], axis=1)
        print('Simulation %d . minimization' %(iteration))
        PATH, NETWORK, COUNTER, seedNN_vector, GENERATION = write_od(Paths, Network_var, OD_min, 'min')
        tmap(RSUMOStateInPar, PATH, NETWORK, COUNTER, seedNN_vector, GENERATION)
        data_simulated = SUMO_aver(Paths, Network_var, 'min')
        y_min = gof_eval_od(data, data_simulated, input_od, OD_min, Network_var['w'])
        rmsn.append(y_min)
        
        print('Iteration NO. %d done' % iteration)
        print('RMSN = ', y_min)
        print('Iterations remaining = %d' % (N-iteration))
        print('========================================')
        for inter in range(len(y_min)):
            if y_min[inter] < Best_RMSN[inter]:
                Best_OD.iloc[:,2+inter] = OD_min.iloc[:,2+inter]
                Best_RMSN[inter] = y_min[inter]
                Best_SimulatedCounts.iloc[:,inter] = data_simulated.iloc[:,inter]
        
    
    print(rmsn)
    end = time.time()
    print('Running time: %d s' %(end-start))
    print('Running time of one simulation: %d s' %(end_one-start_one))
    #%% visualization
    rmsn = pd.DataFrame(rmsn)
    suffix = '100_'+str(method)+str(R[0])+str(R[1])+str(R[2])
    cols = []
    for i in range(len(Network_var["inputOD"])):
        cols.append(str(start_sim+i)+'-'+str(start_sim+i+1))
    rmsn.columns = cols
    plt.rcParams.update({'figure.figsize':(8,8), 'figure.dpi':60, 'figure.autolayout': True})
    plt.figure()
    rmsn.plot.line()
    plt.title('Calibrating Munich with SPSA')
    plt.xlabel("No of Iterations")
    plt.ylabel("RMSN")
    plt.legend()
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('figures'):
        os.mkdir('figures')
    # store the result
    if Network_var['objective'] == 'counts':
        f = open('spatial_temporal/results/method2_counts_rmsn'+suffix+'.pckl', 'wb') # for counts
        plt.savefig('spatial_temporal/figures/method2_rmsn_counts'+suffix+'.pdf')
    if Network_var['objective'] == 'time':
        f = open('spatial_temporal/results/method2_tt_rmsn'+suffix+'.pckl', 'wb') # for travel time
        plt.savefig('spatial_temporal/figures/method2_rmsn_tt'+suffix+'.pdf')
    pickle.dump([rmsn,Best_RMSN,data,Best_SimulatedCounts,input_od,Best_OD], f)
    f.close()
