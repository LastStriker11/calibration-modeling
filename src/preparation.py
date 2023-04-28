# -*- coding: utf-8 -*-
"""
A module for paths (e.g., path to SUMO, path to network) and data (e.g., prior OD estimates, observed traffic measurements) prepartion.

@author: Qing-Long Lu (qinglong.lu@tum.de)
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
        '''
        Initialize a DataPreparation object, with `paths` (a dict of paths to the network, SUMO, scenario and cache) 
        and `sumo_var` (a dict of SUMO simulation configuration) as necessary attributes.

        Parameters
        ----------
        paths : dict
            A dict of paths to SUMO, network, measurements, demand and cache, including:
                <table>
                    <thead>
                        <tr>
                            <th align="left">Variable</th>
                            <th align="left">Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>'sumo'</td>
                            <td>path to the SUMO installation location.</td>
                        </tr>
                        <tr>
                            <td>'network'</td>
                            <td>path to the SUMO network files.</td>
                        </tr>
                        <tr>
                            <td>'demand'</td>
                            <td>path to the prior OD estimates in the [O-format (VISUM/VISSUM)](https://sumo.dlr.de/docs/Demand/Importing_O/D_Matrices.html).</td>
                        </tr>
                        <tr>
                            <td>'measurements'</td>
                            <td>path to the true traffic measurements (in `.csv` format).</td>
                        </tr>
                        <tr>
                            <td>'cache'</td>
                            <td>path to cache folder.</td>
                        </tr>
                    </tbody>
                </table>
        sumo_var : dict
            A dict of SUMO simulation setups, including:
                <table>
                    <thead>
                        <tr>
                            <th align="left">Variable</th>
                            <th align="left">Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>'network'</td>
                            <td>name of the network file.</td>
                        </tr>
                        <tr>
                            <td>'tazname'</td>
                            <td>name of the traffic analysis zone (TAZ) file.</td>
                        </tr>
                        <tr>
                            <td>'add_file'</td>
                            <td>name of the additional file, which includes the detectors information.</td>
                        </tr>
                        <tr>
                            <td>'starttime'</td>
                            <td>when the simulation should start.</td>
                        </tr>
                        <tr>
                            <td>'endtime'</td>
                            <td>when the simulation should stop.</td>
                        </tr>
                        <tr>
                            <td>'objective'</td>
                            <td>indicate the traffic measurements to use, 'counts' or 'time'.</td>
                        </tr>
                        <tr>
                            <td>'interval'</td>
                            <td>calibration interval (in common with the resolution of traffic measurements).</td>
                        </tr>
                    </tbody>
                </table>

        Returns
        -------
        None.

        '''
        self.paths = paths
        self.sumo_var = sumo_var
    
    def load_path(self):
        '''
        Copy necessary files to the cache folder. Simulations and calibration 
        are carried out in this folder to aviod the intermediate data generated by SUMO
        mess up the original network and scenario folders.

        Returns
        -------
        None.

        '''
        self.sumo_var['od_prior'] = [f for f in os.listdir(self.paths['demand'])]
        self.sumo_var['od_prior'].sort()
        try:
            self.sumo_var['truedata'] = [f for f in os.listdir(self.paths['measurements'])]
            self.sumo_var['truedata'].sort()
        except:
            self.sumo_var['truedata'] = 0
        
        # create the folder to store the scenario files
        # if os.path.exists(self.paths["cache"]):
        #     shutil.rmtree(self.paths["cache"])
        #     os.mkdir(self.paths["cache"])
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
        self.sumo_var['endSimTime']   =  str(endSim)
    
    def copy_cache(self):
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
        for file in self.sumo_var["od_prior"]:
            old_loc = self.paths['demand']+file
            new_loc = self.paths['cache']+file
            shutil.copyfile(old_loc, new_loc)
        # truedata
        if self.sumo_var['truedata'] != 0:
            for file in self.sumo_var['truedata']:
                old_loc = self.paths['measurements']+file
                new_loc = self.paths['cache']+file
                shutil.copyfile(old_loc, new_loc)            

    def col_name(self, df):
        start = int(float(self.sumo_var["starttime"]))
        end = int(float(self.sumo_var['endtime']))
        cols = list(range(start, end))
        cols_inter = list(np.arange(start, end, self.sumo_var['interval']))
        df.columns = cols_inter
        return df

    def load_measurement(self):
        
        data = pd.DataFrame()
        # for counts
        if self.sumo_var['objective'] == 'counts':
            for i in range(len(self.sumo_var["truedata"])):
                temp_data = pd.read_csv(self.paths["cache"] + self.sumo_var['truedata'][i], header=None)
                temp_data.set_index(0, inplace=True)
                data = pd.concat([data, temp_data], axis=1)
            data = self.col_name(data)
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
            data = self.col_name(data)
        data.index = data.index.astype(str) # in some networks, it could raise errors
        return data
    
    def load_od(self):
        od_prior = pd.DataFrame()
        for i in range(len(self.sumo_var["od_prior"])):
            temp_od = pd.read_csv(self.paths["cache"] + self.sumo_var["od_prior"][i], sep='\s+', header=None, skiprows=5)
            od_prior = pd.concat([od_prior, temp_od.iloc[:,2]], axis=1)
        od_prior = self.col_name(od_prior)
        od_prior = pd.concat([temp_od.iloc[:,:2], od_prior], axis=1)
        return od_prior
        
    def load_data(self):
        '''
        Load the od_prior estimates and traffic measurements.

        Returns
        -------
        data : TYPE
            DESCRIPTION.
        od_prior : TYPE
            DESCRIPTION.

        '''
        data = self.load_measurement()
        od_prior = self.load_od()
        return data, od_prior
        


        