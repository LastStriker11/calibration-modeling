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
    def __init__(
            self, 
            paths,
            sumo_var
            ):
        self.paths = paths
        self.sumo_var = sumo_var
    
    def load_path(self):
        self.sumo_var['od_prior'] = [f for f in os.listdir(self.paths['network']) if 'start_od' in f]
        self.sumo_var['od_prior'].sort()
        if self.sumo_var['objective'] == 'counts':
            self.sumo_var['truedata'] = os.listdir(self.paths['measurements'])
        if self.sumo_var['objective'] == 'time':
            self.sumo_var['truedata'] = os.listdir(self.paths['measurements'])
        self.sumo_var['truedata'].sort()
        
        # create the folder to store the scenario files
        if os.path.exists(self.paths["cache"]):
            shutil.rmtree(self.paths["cache"])
            os.mkdir(self.paths["cache"])
        if not os.path.exists(self.paths["cache"]):
            os.mkdir(self.paths["cache"])
        #% Start Simulation Times
        integ= np.floor(np.double(self.sumo_var["starttime"]))
        fract=100*(np.double(self.sumo_var["starttime"])-integ)
        beginSim = int(integ*60*60 + fract*60)
        #% End Simulation Times
        integ=np.floor(np.double(self.sumo_var["endtime"]))
        fract=100*(np.double(self.sumo_var["endtime"])-integ)
        endSim = int(integ*60*60 + fract*60)
        
        self.sumo_var['beginSimTime'] =  str(beginSim)
        self.sumo_var['endSimTime']   =  str(endSim+500)
        
        '''
         copy relevant files for SUMO simulation to a folder under the temp directory,
         so that all output files will be located in a same folder which won't make the
         original network folder dirty and give us convenienc to deal with the temporary
         outputs which could be redundant.
        '''
        # network
        old_loc = self.paths['network']+self.sumo_var['network']
        new_loc = self.paths['cache']+self.sumo_var['network']
        shutil.copyfile(old_loc, new_loc)
        # taz
        old_loc = self.paths['network']+self.sumo_var['tazname']
        new_loc = self.paths['cache']+self.sumo_var['tazname']
        shutil.copyfile(old_loc, new_loc)
        # additional
        old_loc = self.paths['network']+self.sumo_var['add_file']
        new_loc = self.paths['cache']+self.sumo_var['add_file']
        shutil.copyfile(old_loc, new_loc)
        # prior od estimates
        for i in range(len(self.sumo_var["od_prior"])):    
            old_loc = self.paths['demand']+self.sumo_var['od_prior'][i]
            new_loc = self.paths['cache']+self.sumo_var['od_prior'][i]
            shutil.copyfile(old_loc, new_loc)
            # truedata
            old_loc = self.paths['measurements']+self.sumo_var['truedata'][i]
            new_loc = self.paths['cache']+self.sumo_var['truedata'][i]
            shutil.copyfile(old_loc, new_loc)            
        
    def load_data(self):
        start = int(float(self.sumo_var["starttime"]))
        end = int(float(self.sumo_var['endtime']))
        cols = list(range(start, end))
        cols_inter = list(np.arange(start, end, self.sumo_var['interval']))
        data = pd.DataFrame()
        # for counts
        if self.sumo_var['objective'] == 'counts':
            for i in range(len(self.sumo_var["truedata"])):
                temp_data = pd.read_csv(self.paths["cache"] + self.sumo_var['truedata'][i], header=None)
                temp_data.set_index(0, inplace=True)
                data = pd.concat([data, temp_data], axis=1)
            data.columns = cols_inter
            #@@@@
            data = data[(data.T!=0).any()]
            #@@@@
        # for travel time
        if self.sumo_var['objective'] == 'time':
            for i in range(len(self.sumo_var["truedata"])):
                temp_data = pd.read_csv(self.paths["cache"] + self.sumo_var['truedata'][i], header=None)
                temp_data['label'] = list(zip(temp_data.iloc[:,0], temp_data.iloc[:,1]))
                temp_data.drop([0,1], axis=1, inplace=True)
                temp_data.set_index('label', inplace=True)
                data = pd.concat([data, temp_data], axis=1)
            data.columns = cols
        # prior od estimates
        od_prior = pd.DataFrame()
        for i in range(len(self.sumo_var["od_prior"])):
            temp_od = pd.read_csv(self.paths["cache"] + self.sumo_var["od_prior"][i], sep='\s+', header=None, skiprows=5)
            od_prior = pd.concat([od_prior, temp_od.iloc[:,2]], axis=1)
        od_prior.columns = cols_inter
        od_prior = pd.concat([temp_od.iloc[:,:2], od_prior], axis=1)
        return data, od_prior


        