# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:06:52 2019

@author: lasts
"""
import pandas as pd
import numpy as np
import Basic_scripts as bs
from pathos.multiprocessing import ProcessingPool as PPool
from pathos.multiprocessing import ThreadingPool as TPool
import pickle
import matplotlib.pyplot as plt
import time
from Functions import load_path, load_data, write_od, RSUMOStateInPar, SUMO_aver, gof_eval, scenario_import
from Functions import generation_SPSA as generation
from Functions import rep_generation_SPSA as rep_generation
#%%
if __name__ == '__main__':
    user='qinglong'
    start = time.time()
    # load path and data
    Paths, Network_var, SPSA_para, PC_SPSA_para = load_path(user)
    Paths = scenario_import(Paths, Network_var)
    data, input_od, hist_od = load_data(Paths, Network_var)
    convergence = SPSA_para["Min_error"]
    a = SPSA_para["a"]       
    c = SPSA_para["c"]
    A = SPSA_para["A"]
    alpha = SPSA_para["alpha"]
    gamma = SPSA_para["gamma"]
    G = SPSA_para["G"]
    N = SPSA_para["N"]
    seg = SPSA_para['seg']
    #%% the starting simulation
    ODbase = bs.Vector2Full(input_od.iloc[:,2])
    OD = ODbase.copy()
    Best_OD = input_od.iloc[:,2]
    Best_RMSN = 100
    Best_SimulatedCounts = 0
    OD_plus = input_od.copy()
    OD_minus = input_od.copy()
    OD_min = input_od.copy()
    rmsn = []
    list_ak = []
    list_ck = []
    list_g = []
    #%%
    pamap = PPool().amap # asynchronous processing
    tmap = TPool().map # synachronous processing
    print('Simulation 0 started')
    PATH, NETWORK, COUNTER, seedNN_vector, GENERATION = write_od(Paths, Network_var, input_od, 'start')
    tmap(RSUMOStateInPar, PATH, NETWORK, COUNTER, seedNN_vector, GENERATION)
    data['simulated'] = SUMO_aver(Paths, Network_var, 'start')
    data_eval = data.dropna(axis=0)
    print('Simultaion 0 completed')
    y = gof_eval(data_eval)
    rmsn.append(y)
    print('Starting RMSN = ', y)
    print('========================================')
    #%%
    for iteration in range(1, N + 1):
        # calculating gain sequence parameters
        ak = a / ((iteration + A) ** alpha)
        ck = c / (iteration ** gamma)
        list_ak.append(ak)
        list_ck.append(ck)
        m = np.mean(OD)
        
        GA, INPUT_OD, OD_BASE, ODS, SEG, CK, OD_PLUS, OD_MINUS, DATA, PATH, NETWORK = rep_generation(G, input_od, ODbase, OD, seg, ck, OD_plus, OD_minus, data, Paths, Network_var)
        # the 'outer' parallel processing
        g_hat_it = np.stack(pamap(generation, GA, INPUT_OD, OD_BASE, ODS, SEG, CK, OD_PLUS, \
                                      OD_MINUS, DATA, PATH, NETWORK, [tmap]*G).get(), axis=-1)
        g_hat_it = pd.DataFrame(g_hat_it)
        g_hat = g_hat_it.mean(axis=1)
#        g_hat_mean = g_hat.mean()
#        if iteration == 1:
#            ak = p_h * ck / g_hat_mean
#            a = ak * ((iteration + A) ** alpha)
        list_g.append(abs(g_hat).mean())
        for i in range(1, int(np.fix(OD.max()/seg)) + 1):#!!
            for f in range(0, OD.shape[1]):
                for e in range(0, OD.shape[0]):
                    if OD[e, f] > 4:
                        q = i * seg
                        p = q - seg
                        if OD[e, f] > p and OD[e, f] <= q:
                            if OD[e, f] == ODbase[e, f]:
                                OD[e, f] = OD[e, f] - ((ak * g_hat[e*OD.shape[0]+f] * q) / m)
                            diff = ((OD[e, f] - ODbase[e, f]) / ODbase[e, f])
                            if diff < -0.15:
                                OD[e, f] = ODbase[e, f] * 0.85
                            if diff > 0.15:
                                OD[e, f] = ODbase[e, f] * 1.15
        ODbase = OD.copy()
        OD_min.iloc[:,2] = pd.DataFrame(OD).stack().values#!!
#        allODVectors[:,iteration-1] = OD[:,0]
        print('Simulation %d . minimization' %(iteration))
        PATH, NETWORK, COUNTER, seedNN_vector, GENERATION = write_od(Paths, Network_var, OD_min, 'min')
        tmap(RSUMOStateInPar, PATH, NETWORK, COUNTER, seedNN_vector, GENERATION)
        data['simulated'] = SUMO_aver(Paths, Network_var, 'min')
        data_eval = data.dropna(axis=0)
        y_min = gof_eval(data_eval)
        
        rmsn.append(y_min)
        
        print('Iteration NO. %d done' % iteration)
        print('RMSN = ', y_min)
        print('Iterations remaining = %d' % (N-iteration))
        print('========================================')
        if y_min < Best_RMSN:
            Best_OD = OD_min.iloc[:,2]
            Best_RMSN = y_min
            Best_SimulatedCounts = data['simulated']
        if y_min < convergence:
            break
    if Network_var['truedata'] == 'true_counts.csv':
        f = open('counts_SPSA.pckl', 'wb') # for counts
    if Network_var['truedata'] == 'true_tt.csv':
        f = open('tt_SPSA.pckl', 'wb') # for travel time
    pickle.dump([Best_RMSN, Best_OD, Best_SimulatedCounts, rmsn, list_ak, list_ck, list_g], f)
    f.close()
    end = time.time()
    print(rmsn)
    #%% visualization
    plt.rcParams.update({'figure.figsize':(8,8), 'figure.dpi':60, 'figure.autolayout': True})
    plt.figure()
    plt.title('Calibrating Munich with SPSA')
    plt.plot(rmsn, label = 'SPSA')
    plt.xlabel("No of Iterations")
    plt.ylabel("RMSN")
    plt.legend()
    plt.show()
    plt.savefig('../figures/rmsn.pdf')
    print('Running time: %d s' %(end-start))


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    