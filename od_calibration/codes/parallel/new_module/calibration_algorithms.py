# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 09:14:00 2022

@author: lasts
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from pathos.multiprocessing import ProcessingPool as PPool
from pathos.multiprocessing import ThreadingPool as TPool

from sumo_operation import SUMOOperation
from evaluation_metrics import normalized_root_mean_square as rmsn

class PC_SPSA(SUMOOperation):
    def __init__(self, paths, sumo_var, paras, od_prior, data_true):
        self.paras = paras
        self.od_prior = od_prior
        self.data_true = data_true
        super().__init__(paths=paths, sumo_var=sumo_var)
    
    def pc_spsa_perturb(self, ga, ck, n_compon, z, V, index_same, tmap):
        start_sim = int(float(self.sumo_var["starttime"]))
        # build different folders for different generation
        if not os.path.exists(self.paths['cache']+str(ga)):
            os.makedirs(self.paths['cache']+str(ga))
        self.paths['cache'] = self.paths['cache']+str(ga)+'/'
        
        Deltas = []
        OD_plus = pd.DataFrame()
        for i in range(len(self.sumo_var["inputOD"])):
            
            delta = 2*np.random.binomial(n=1, p=0.5, size=n_compon[i])-1 # Bernoulli distribution
            # plus perturbation
            z_per = z[i] + z[i]*ck*delta
            temp_per = pd.DataFrame(np.matmul(V[i],z_per))
            temp_per = pd.concat([self.od_prior.iloc[:,:2], temp_per], axis=1)
            temp_per.columns = ['from', 'to', 'counts']
            temp_per.loc[index_same.iloc[:,i], 'counts'] = self.od_prior.loc[index_same.iloc[:,i], start_sim+i*self.sumo_var['interval']]
            index_neg = temp_per.loc[:,'counts']<3
            temp_per.loc[index_neg, 'counts'] = self.od_prior.loc[index_neg, start_sim+i*self.sumo_var['interval']]
            OD_plus = pd.concat([OD_plus, temp_per['counts']], axis=1)
            Deltas.append(delta)
        OD_plus = pd.concat([self.od_prior.iloc[:,:2], OD_plus], axis=1)
        COUNTER, seedNN_vector, GENERATION = self.write_od(OD_plus, ga)
        tmap(self.sumo_run, COUNTER, seedNN_vector, GENERATION)
        data_simulated = self.SUMO_aver(ga)
        y_plus = rmsn(self.data_true, data_simulated, self.od_prior, OD_plus, self.sumo_var['w'])
        
        # minus perturbation
        OD_minus = pd.DataFrame()
        for i in range(len(self.sumo_var["inputOD"])):
            delta = Deltas[i] # Bernoulli distribution
            # plus perturbation
            z_per = z[i] - z[i]*ck*delta
            temp_per = pd.DataFrame(np.matmul(V[i],z_per))
            temp_per = pd.concat([self.od_prior.iloc[:,:2], temp_per], axis=1)
            temp_per.columns = ['from', 'to', 'counts']
            temp_per.loc[index_same.iloc[:,i], 'counts'] = self.od_prior.loc[index_same.iloc[:,i], start_sim+i*self.sumo_var['interval']]
            index_neg = temp_per.loc[:,'counts']<3
            temp_per.loc[index_neg, 'counts'] = self.od_prior.loc[index_neg, start_sim+i*self.sumo_var['interval']]
            OD_minus = pd.concat([OD_minus, temp_per['counts']], axis=1)
        OD_minus = pd.concat([self.od_prior.iloc[:,:2], OD_minus], axis=1)
        COUNTER, seedNN_vector, GENERATION = self.write_od(OD_minus, ga)
        tmap(self.sumo_run, COUNTER, seedNN_vector, GENERATION)
        data_simulated = self.SUMO_aver(self.paths, self.sumo_var, ga)
    #    data_simulated = data_simulated.dropna(axis=0)
        y_minus = rmsn(self.data_true, data_simulated, self.od_prior, OD_minus, self.sumo_var['w'])
        
        # Gradient Evaluation
        g_hat = pd.DataFrame()
        for i in range(len(self.sumo_var["inputOD"])):
            temp_g = (y_plus[i] - y_minus[i])/(2*ck*Deltas[i])
            g_hat = pd.concat([g_hat, pd.DataFrame(temp_g)], axis=1)
        return g_hat
    
    def rep_generation(self, ck, n_compon, z, V, index_same):
        GA = list(range(0, self.sumo_var['n_gen']))
        CK = []
        N_COMPON = []
        ZS = []
        VS = []
        INDEX_SAME = []
        for num in range(self.sumo_var['n_gen']):
            CK.append(ck)
            N_COMPON.append(n_compon)
            ZS.append(z)
            VS.append(V)
            INDEX_SAME.append(index_same)
        return GA, CK, N_COMPON, ZS, VS, INDEX_SAME
    
    def initial_setup(self):
        V = []
        z = []
        n_compon = []
        temp_U, temp_S, temp_V = np.linalg.svd(self.hist_od, full_matrices=False)
        temp_cv = temp_S.cumsum()/temp_S.sum()
        for temp_compon, score in enumerate(temp_cv): # find n_compon which can lead to a score > 0.95
            if score > self.paras['variance']:
                break
        temp_V = temp_V[:temp_compon,:]
        for i in range(len(self.sumo_var["inputOD"])):
            temp_z = np.matmul(temp_V, self.od_prior.iloc[:,i+2].values)
            V.append(temp_V.T)
            z.append(temp_z)
            n_compon.append(temp_compon)
            
        best_od = self.od_prior.copy()
        best_metrics = [100]*(self.od_prior.shape[1]-2)
        best_data = self.data_true.copy()
        index_same = self.od_prior.iloc[:,2:]<5
        return V, z, n_compon, best_od, best_metrics, best_data, index_same
    
    def create_history(self, hist_method, R, hist_days=100):
        '''
        Create a historical OD matrix for implementing the PC-SPSA algorithm for network/OD calibration.
    
        Parameters
        ----------
        hist_method : int
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
        if hist_method == 1 or hist_method == 2: 
            for i in range(self.od_prior.shape[1]-2): # iterate over time intervals
                od = self.od_prior.iloc[:,[0,1,i+2]]
                if hist_method == 1:
                    for d in range(hist_days):
                        delta = np.random.normal(0, 0.333, size=(od.shape[0]))
                        delta = delta.reshape(od.shape[0],1)
                        if d == 0: 
                            deltas = delta
                        else:
                            deltas = np.concatenate((deltas,delta),axis=1)
                elif hist_method == 2:
                    deltas = np.random.normal(0, 0.333, size=(od.shape[0], hist_days))   # one method without normal distribution hist_days
                # dm formulation with OD pair colrrelation
                factor = R[0]*deltas + np.ones(shape=(od.shape[0], hist_days))
                temp_dm = pd.DataFrame(np.multiply(factor.T, od.iloc[:,2].values))
                temp_dm = temp_dm.T
                D_m = pd.concat([D_m, temp_dm], axis=1) # check axis
            D_m = D_m.T
        
        # method 3 Temporal correlation 
        # method 4 Temporal + days correlation
        if hist_method == 3 or hist_method == 4:        
            if hist_method == 4:
                D = np.random.normal(0.5, 0.08325, size=(hist_days))
            else:
                D = [1] * hist_days
            for d in D:
                for i in range(self.od_prior.shape[0]): # iterate over OD pairs
                    delta = np.random.normal(0, 0.333, size=(self.od_prior.shape[1]-2))
                    delta = delta.reshape(self.od_prior.shape[1]-2,1)
                    if i == 0: 
                        deltas = delta
                    else:
                        deltas = np.concatenate ((deltas, delta),axis=1)       
                deltas = deltas.T
                factor = R[1]*d*deltas + np.ones(shape=(self.od_prior.shape[0], self.od_prior.shape[1]-2))
                temp_dm = factor*self.od_prior.iloc[:,2:].values
                temp_dm = pd.DataFrame(temp_dm)
                temp_dm = temp_dm.T
                D_m = pd.concat([D_m, temp_dm], axis=0) # check axis
            
            temp_dm = temp_dm.T
    
        # method 5 Spatial + temporal correlation
        # method 6 Spatial + temporal + days correlation
                
        if hist_method == 5 or hist_method == 6: # create a rand_matrix with multiple normal distributions
            if hist_method == 6:
                D = np.random.normal(0.5, 0.08325, size=(hist_days))
            else:
                D = [1] * hist_days
            for d in D:
                deltas = np.random.normal(0, 0.333, size=(self.od_prior.shape[0],self.od_prior.shape[1]-2))
                factor = min(R[:2])*d*deltas + np.ones(shape=(self.od_prior.shape[0], self.od_prior.shape[1]-2))
                temp_dm = factor*self.od_prior.iloc[:,2:].values
                temp_dm = pd.DataFrame(temp_dm)
                temp_dm = temp_dm.T
                D_m = pd.concat([D_m, temp_dm], axis=0) # check axis
            
            temp_dm = temp_dm.T
                
        return D_m
    
    def run(self):
        start = time.time()
        start_sim = int(float(self.sumo_var["starttime"]))
        V, z, n_compon, best_od, best_metrics, best_data, index_same = self.initial_setup()
        convergence = [] # to store the metrics of each iteration
        
        n_sumo = self.sumo_var['n_sumo']
        n_gen = self.sumo_var['n_gen']
        start_one = time.time()
        
        pamap = PPool(self.paras['n_gen']).amap # asynchronous processing
        tmap = TPool(n_sumo).map # synachronous processing
        COUNTER, seedNN_vector, GENERATION = self.write_od(self.paths, self.sumo_var, self.od_prior, 'start')
        
        tmap(self.sumo_run, COUNTER, seedNN_vector, GENERATION)
        #=============================='''
        data_simulated = self.SUMO_aver(self.paths, self.sumo_var, 'start')
    #    data_simulated.dropna(axis=0, inplace=True)
        print('Simultaion 0 completed')
        y = rmsn(self.data_true, data_simulated)
        convergence.append(y)
        end_one = time.time()
        print('Starting RMSN = ', y)
        print('========================================')

        for iteration in range(1, self.n_iter + 1):
            # calculating gain sequence parameters
            ak = self.paras['a'] / ((iteration + self.paras['A']) ** self.paras['alpha'])
            ck = self.paras['c'] / (iteration ** self.paras['gamma'])
            
            GA, CK, N_COMPON, ZS, VS, INDEX_SAME = self.rep_generation(n_gen, ck, n_compon, z, V, index_same)
            # the 'outer' parallel processing
            G_hat = np.stack(pamap(self.pc_spsa_perturb, GA, CK, N_COMPON, ZS, VS, \
                                      INDEX_SAME, [tmap]*n_gen).get(), axis=-1)
            
            g_hat_it = np.zeros((max(n_compon), len(self.sumo_var["inputOD"])))
            for i in range(n_gen):
                temp_g = pd.DataFrame(G_hat[:,:,i])
                g_hat_it = g_hat_it + temp_g
            g_hat_it = pd.DataFrame(g_hat_it/n_gen).T
            
            # minimization
            OD_min = pd.DataFrame()
            for i in range(len(self.sumo_var["inputOD"])):
                g_hat = g_hat_it.iloc[i,:].dropna()
                z_per = z[i] - np.multiply(z[i],(ak*g_hat.values))
                temp_per = pd.DataFrame(np.matmul(V[i],z_per)) # temp per is the OD minimzied
                temp_per = pd.concat([self.od_prior.iloc[:,:2], temp_per], axis=1)
                temp_per.columns = ['from', 'to', 'counts']
                temp_per.loc[index_same.iloc[:,i], 'counts'] = self.od_prior.loc[index_same.iloc[:,i], start_sim+i*self.sumo_var['interval']] # index_same is the index of ODs less than 5
                index_neg = temp_per.loc[:,'counts']<3
                temp_per.loc[index_neg, 'counts'] = self.od_prior.loc[index_neg, start_sim+i*self.sumo_var['interval']]
                OD_min = pd.concat([OD_min, temp_per['counts']], axis=1)
                z[i] = z_per.copy()
                
            OD_min = pd.concat([self.od_prior.iloc[:,:2], OD_min], axis=1)
            print('Simulation %d . minimization' %(iteration))
            PATH, NETWORK, COUNTER, seedNN_vector, GENERATION = self.write_od(self.paths, self.sumo_var, OD_min, 'min')
            tmap(self.sumo_run, COUNTER, seedNN_vector, GENERATION)
            data_simulated = self.SUMO_aver(self.paths, self.sumo_var, 'min')
            y_min = rmsn(self.data_true, data_simulated, self.od_prior, OD_min, self.sumo_var['w'])
            convergence.append(y_min)
            
            print('Iteration NO. %d done' % iteration)
            print('RMSN = ', y_min)
            print('Iterations remaining = %d' % (self.n_iter-iteration))
            print('========================================')
            for inter in range(len(y_min)):
                if y_min[inter] < best_metrics[inter]:
                    best_od.iloc[:,2+inter] = OD_min.iloc[:,2+inter]
                    best_metrics[inter] = y_min[inter]
                    best_data.iloc[:,inter] = data_simulated.iloc[:,inter]
                    
        print(convergence)
        end = time.time()
        print('Running time: %d s' %(end-start))
        print('Running time of one simulation: %d s' %(end_one-start_one))

        convergence = pd.DataFrame(convergence)
        cols = []
        for i in range(len(self.sumo_var["inputOD"])):
            cols.append(str(start_sim+i)+'-'+str(start_sim+i+1))
        convergence.columns = cols
    
        plt.rcParams.update({'figure.figsize':(8,8), 'figure.dpi':60, 'figure.autolayout': True})
        plt.figure()
        convergence.plot.line()
        plt.title('Convergence plot')
        plt.xlabel("Iterations")
        plt.ylabel("RMSN")
        plt.legend()
        
        return convergence, best_metrics,best_data,best_od
