# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:43:21 2022

@author: lasts
"""

import pandas as pd
import numpy as np
import os
#%%
def create_Dm(Paths, Network_var, input_od, method, R, hist_days=100):
    '''
    This function is used to create a historical OD matrix for implementing the PC-SPSA algorithm for network/OD calibration.

    Parameters
    ----------
    Paths : dictionary
        Paths to the dependencies.
    Network_var : dictionary
        SUMO network files and simulation configurations.
    input_od : DataFrame
        Prior OD estimates.
    method : int
        Method to generate the historical OD matrix.
    R : list (three elements)
        distribution factors for spatial, temporal and days correlations, respectively.
    hist_days: int
        Dimension of the historical OD matrix, i.e., how many days.

    Returns
    -------
    D_m : DataFrame
        The historical OD matrix.

    '''
    D_m = pd.DataFrame()
    np.random.seed(1)
    # method 1 Spatial correlation 
    # method 2 Spatial + days correlation
    if method == 1 or method == 2: 
        for i in range(input_od.shape[1]-2): # iterate over time intervals
            od = input_od.iloc[:,[0,1,i+2]]
            if method == 1:
                for d in range(hist_days):
                    delta = np.random.normal(0, 0.333, size=(od.shape[0]))
                    delta = delta.reshape(od.shape[0],1)
                    if d == 0: 
                        deltas = delta
                    else:
                        deltas = np.concatenate((deltas,delta),axis=1)
            elif method == 2:
                deltas = np.random.normal(0, 0.333, size=(od.shape[0], hist_days))   # one method without normal distribution hist_days
            # dm formulation with OD pair colrrelation
            factor = R[0]*deltas + np.ones(shape=(od.shape[0], hist_days))
            temp_dm = pd.DataFrame(np.multiply(factor.T, od.iloc[:,2].values))
            temp_dm = temp_dm.T
            # temp_dm.T.to_csv(Paths["origin_network"] + folder+'/' +Network_var['inputOD'][i][:-4]+'temp_dm.csv', index=False, header=False)
            D_m = pd.concat([D_m, temp_dm], axis=1) # check axis
        D_m = D_m.T
    
    # method 3 Temporal correlation 
    # method 4 Temporal + days correlation
    if method == 3 or method == 4:        
        if method == 4:
            D = np.random.normal(0.5, 0.08325, size=(hist_days))
        else:
            D = [1] * hist_days
        for d in D:
            for i in range(input_od.shape[0]): # iterate over OD pairs
                delta = np.random.normal(0, 0.333, size=(input_od.shape[1]-2))
                delta = delta.reshape(input_od.shape[1]-2,1)
                if i == 0: 
                    deltas = delta
                else:
                    deltas = np.concatenate ((deltas, delta),axis=1)       
            deltas = deltas.T
            factor = R[1]*d*deltas + np.ones(shape=(input_od.shape[0], input_od.shape[1]-2))
            temp_dm = factor*input_od.iloc[:,2:].values
            temp_dm = pd.DataFrame(temp_dm)
            temp_dm = temp_dm.T
            D_m = pd.concat([D_m, temp_dm], axis=0) # check axis
        
        temp_dm = temp_dm.T

    # method 5 Spatial + temporal correlation
    # method 6 Spatial + temporal + days correlation
            
    if method == 5 or method == 6: # create a rand_matrix with multiple normal distributions
        if method == 6:
            D = np.random.normal(0.5, 0.08325, size=(hist_days))
        else:
            D = [1] * hist_days
        for d in D:
            deltas = np.random.normal(0, 0.333, size=(input_od.shape[0],input_od.shape[1]-2))
            factor = min(R[:2])*d*deltas + np.ones(shape=(input_od.shape[0], input_od.shape[1]-2))
            temp_dm = factor*input_od.iloc[:,2:].values
            temp_dm = pd.DataFrame(temp_dm)
            temp_dm = temp_dm.T
            D_m = pd.concat([D_m, temp_dm], axis=0) # check axis
        
        temp_dm = temp_dm.T
            
    return D_m