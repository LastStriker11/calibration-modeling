# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 21:19:12 2022

@author: lasts
"""
import pandas as pd
import numpy as np
import os
import shutil

class DataPreparation(object):
    # network, scenario=False, objective='counts', method=0, NSumoRep=2, G=1, N=2, self.x=None, y=None
    def __init__(
            self, 
            paths,
            net_var,
            network, 
            scenario=False, 
            objective='count', 
            method=6, 
            n_sumo=2, 
            n_gen=1, 
            n_iter=2,
            x=None,
            y=None
            ):
        self.paths = paths
        self.net_var = net_var
        self.network = network
        self.scenario = scenario
        self.objective = objective
        self.method = method
        self.n_sumo = n_sumo
        self.n_gen = n_gen
        self.n_iter = n_iter
        self.x = x
        self.y = y
    
    def load_path(self):
        self.net_var['flag'] = (self.scenario)|(self.x is None)
        scenario_type = 'spatial_temporal'
        if (self.scenario == True) or (self.x is None):
            folder = 'Demand'
            self.net_var['inputOD'] = [f for f in os.listdir(self.paths['pathToNetwork_a']+folder) if f.endswith('.txt')]  
            self.net_var['inputOD'].sort()
            
        if (self.scenario == False) & (self.x is not None):
            self.net_var['para'] = 'scenarios/'+scenario_type+'/'+str(self.x)+'_'+str(self.y)
            folder = 'D_MandStartOD/'+str(self.method)
            self.net_var['inputOD'] = [f for f in os.listdir(self.paths['pathToNetwork_a']+folder) if 'start_od' in f]
            self.net_var['inputOD'].sort()
            if self.net_var['objective'] == 'counts':
                self.net_var['truedata'] = os.listdir(self.paths['pathToNetwork_a']+self.net_var['para']+'/counts')
            if self.net_var['objective'] == 'time':
                self.net_var['truedata'] = os.listdir(self.paths['pathToNetwork_a']+self.net_var['para']+'/time')
            self.net_var['truedata'].sort()
        
        # create the folder to store the scenario files
        self.paths["pathToScenario"] = self.paths["tempLocalLocation"]+self.net_var["scenarioID"]
        if os.path.exists(self.paths["pathToScenario"]):
            shutil.rmtree(self.paths["pathToScenario"])
            os.mkdir(self.paths["pathToScenario"])
        if not os.path.exists(self.paths["pathToScenario"]):
            os.mkdir(self.paths["pathToScenario"])
        self.paths["pathToScenario"] = self.paths["pathToScenario"]+str(self.method)+'/'
        if os.path.exists(self.paths["pathToScenario"]):
            shutil.rmtree(self.paths["pathToScenario"])
            os.mkdir(self.paths["pathToScenario"])
        if not os.path.exists(self.paths["pathToScenario"]):
            os.mkdir(self.paths["pathToScenario"])
        #% Start Simulation Times
        integ= np.floor(np.double(self.net_var["starttime"]))
        fract=100*(np.double(self.net_var["starttime"])-integ)
        beginSim = int(integ*60*60 + fract*60)
        #% End Simulation Times
        integ=np.floor(np.double(self.net_var["endtime"]))
        fract=100*(np.double(self.net_var["endtime"])-integ)
        endSim = int(integ*60*60 + fract*60)
        
        self.net_var['beginSimTime'] =  str(beginSim)
        self.net_var['endSimTime']   =  str(endSim+500)
        
        '''
         copy relevant files for SUMO simulation to a folder under the temp directory,
         so that all output files will be located in a same folder which won't make the
         original network folder dirty and give us convenienc to deal with the temporary
         outputs which could be redundant.
        '''
        # mesoNet
        old_loc = self.paths['pathToNetwork_a']+self.net_var['mesoNet']
        new_loc = self.paths['pathToScenario']+self.net_var['mesoNet']
        shutil.copyfile(old_loc, new_loc)
        # taz
        old_loc = self.paths['pathToNetwork_a']+self.net_var['tazname']
        new_loc = self.paths['pathToScenario']+self.net_var['tazname']
        shutil.copyfile(old_loc, new_loc)
        # additional
        old_loc = self.paths['pathToNetwork_a']+self.net_var['additionalName']
        new_loc = self.paths['pathToScenario']+self.net_var['additionalName']
        shutil.copyfile(old_loc, new_loc)
        # inputOD
        for i in range(len(self.net_var["inputOD"])):    
            old_loc = self.paths['pathToNetwork_a']+folder+'/'+self.net_var['inputOD'][i]
            new_loc = self.paths['pathToScenario']+self.net_var['inputOD'][i]
            shutil.copyfile(old_loc, new_loc)
            if self.net_var['flag'] == False:
                # truedata
                old_loc = self.paths['pathToNetwork_a']+self.net_var['para']+'/'+self.net_var['objective']+'/'+self.net_var['truedata'][i]
                new_loc = self.paths['pathToScenario']+self.net_var['truedata'][i]
                shutil.copyfile(old_loc, new_loc)            
        # change the network folder to the temporal position
        self.paths['origin_network'] = self.paths['pathToNetwork_a']
        self.paths['pathToNetwork_a'] = self.paths['pathToScenario']
        
        
def create_history(self, input_od, method, R, hist_days=100):
    '''
    Create a historical OD matrix for implementing the PC-SPSA algorithm for network/OD calibration.

    Parameters
    ----------
    net_var : dictionary
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
    if self.method == 1 or self.method == 2: 
        for i in range(input_od.shape[1]-2): # iterate over time intervals
            od = input_od.iloc[:,[0,1,i+2]]
            if self.method == 1:
                for d in range(hist_days):
                    delta = np.random.normal(0, 0.333, size=(od.shape[0]))
                    delta = delta.reshape(od.shape[0],1)
                    if d == 0: 
                        deltas = delta
                    else:
                        deltas = np.concatenate((deltas,delta),axis=1)
            elif self.method == 2:
                deltas = np.random.normal(0, 0.333, size=(od.shape[0], hist_days))   # one method without normal distribution hist_days
            # dm formulation with OD pair colrrelation
            factor = R[0]*deltas + np.ones(shape=(od.shape[0], hist_days))
            temp_dm = pd.DataFrame(np.multiply(factor.T, od.iloc[:,2].values))
            temp_dm = temp_dm.T
            D_m = pd.concat([D_m, temp_dm], axis=1) # check axis
        D_m = D_m.T
    
    # method 3 Temporal correlation 
    # method 4 Temporal + days correlation
    if self.method == 3 or self.method == 4:        
        if self.method == 4:
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
            
    if self.method == 5 or self.method == 6: # create a rand_matrix with multiple normal distributions
        if self.method == 6:
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
        