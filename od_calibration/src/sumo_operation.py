# -*- coding: utf-8 -*-
"""
Include the functions for reading SUMO inputs, writing outputs, running simulations.

@author: Qing-Long Lu (qinglong.lu@tum.de)
"""
import pandas as pd
import numpy as np
from preparation import DataPreparation
import subprocess

class SUMOOperation(DataPreparation):
    def __init__(self, paths, sumo_var):
        '''
        Initialize a SUMOOperation object. This class inherits the functions from the DataPreparation class.

        Parameters
        ----------
        paths : dict
            A dict of paths to SUMO, network, measurements, demand and cache.
        sumo_var : dict
            A dict of SUMO simulation setups.

        Returns
        -------
        None.

        '''
        super().__init__(paths=paths, sumo_var=sumo_var)

    def write_od(self, od_matrix, ga=0):
        '''
        Convert the OD matrix from DataFrame format to SUMO format.

        Parameters
        ----------
        od_matrix : DataFrame
            The OD matrix to write.
        ga : int or string, optional
            An indicator of gradient sample.

        Returns
        -------
        COUNTER : list
            A list of indictors for simulation index (length is `n_sumo`) 
            for distinguishing different simulation instance in parallel computing.
        seedNN_vector : array
            An array of random seeds for simulation (length is `n_sumo`) 
            for generating different simulation outputs in parallel computing.
        GENERATION : list
            A list of `ga` (length is `n_sumo`).

        '''
        start = int(float(self.sumo_var["starttime"]))
        end = int(float(self.sumo_var['endtime']))
        cols = list(np.arange(start, end, self.sumo_var['interval']))
        cols = [0, 1] + cols
        od_matrix.columns = cols
        for i in range(od_matrix.shape[1]-2):
            tt    = start+self.sumo_var['interval']*i
            inter = int(tt)
            fract = (tt - inter)*60
            if fract == 0:
                TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.00 ' 
                               + str(inter)+'.' + str(int(fract+60*self.sumo_var['interval'])) + '\n* Factor \n1.00\n')
            if fract == 45:
                TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.' + str(45)+' ' 
                               + str(inter+1)+ '.00' + '\n* Factor \n1.00\n')
            else:
                TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.' + str(int(fract))+' ' 
                               + str(inter)+'.' + str(int(fract+60*self.sumo_var['interval'])) + '\n* Factor \n1.00\n')
            ma = self.paths["cache"] + 'OD_updated_'+str(od_matrix.columns[i+2])+'.txt'
            file = open(ma, 'w')
            file.write(TheBeginText)
            file.close()
            od_matrix.iloc[:, [0,1,i+2]].to_csv(ma, header=False, index=False, sep=' ', mode='a')
        # replicate the variables
        seedNN_vector = np.random.normal(0, 10000, self.sumo_var["n_sumo"]).tolist()
        COUNTER = list(range(0,self.sumo_var['n_sumo']))
        GENERATION =[]
        for num in range(self.sumo_var['n_sumo']):
            GENERATION.append(ga)
        return COUNTER, seedNN_vector, GENERATION
    
    def sumo_run(self, k, seed, ga):
        '''
        Run sumo simulation including: `od2trips` and `sumo`.

        Parameters
        ----------
        k : int
            An indicator of simulation instance.
        seed : int
            A random value.
        ga : int or string
            An indicator of gradient sample.

        Returns
        -------
        None.

        '''
        start = int(float(self.sumo_var["starttime"]))
        end = int(float(self.sumo_var['endtime']))
        rerouteProb = 0.1
        cols = list(np.arange(start, end, self.sumo_var['interval']))
        # OD2TRIPS, generate upd_ODTrips.trips.xml
        prefx = str(ga)+'_'+str(k)+'_'
        seed = int(seed)
        oD2TRIPS = (self.paths["sumo"]+'bin/od2trips --no-step-log --output-prefix '+prefx+ 
                    ' --spread.uniform --taz-files '+ self.paths["cache"] + self.sumo_var["tazname"] +' -d ')
            
        for i in range(len(cols)):
            oD2TRIPS = oD2TRIPS + self.paths["cache"] + 'OD_updated_'+str(cols[i])+'.txt,'
        oD2TRIPS = oD2TRIPS[:-1]
        oD2TRIPS = oD2TRIPS + ' -o '+ self.paths["cache"]+'upd_ODTrips.trips.xml --seed '+str(seed)
        '''
        If passing a single string, either shell must be True (see below) or else the string must simply 
        name the program to be executed without specifying any arguments.
        '''
        subprocess.run(oD2TRIPS, shell=True)
        
        # generate out.xml, vehroutes.xml
        sumoRUN = (self.paths["sumo"]+'bin/sumo --mesosim  --no-step-log -W --output-prefix '+
                   prefx+' -n '+self.paths["cache"]+self.sumo_var["network"]+
                   ' -b '+self.sumo_var['beginSimTime']+
                   ' -e '+ self.sumo_var['endSimTime']+
                   ' -r '+self.paths["cache"]+prefx+'upd_ODTrips.trips.xml'+
                   ' --vehroutes ' + self.paths["cache"] + 'network.vehroutes.xml'+
                   ' --additional-files '+ self.paths["cache"] + self.sumo_var["add_file"] +
                   ' --xml-validation never'+
                   ' --device.rerouting.probability '+str(rerouteProb)+
                   ' --seed '+str(seed))
        subprocess.run(sumoRUN, shell=True)
        
    def sumo_aver(self, ga):
        '''
        Aggregate the simulation outputs.

        Parameters
        ----------
        ga : int
            An indicator of gradient sample.

        Returns
        -------
        measurements_agg : DataFrame
            Simulation ouputs after proper aggregration based on the calibration interval.

        '''
        
        fract = float(self.sumo_var["starttime"])
        start = int(fract)
        end = int(float(self.sumo_var['endtime']))
        fract = round(fract-start, 2)
        agg_inter = self.sumo_var['interval']
        # traffic counts
        if self.sumo_var['objective'] == 'counts':
            counts = pd.DataFrame()
            for counter in range(self.sumo_var['n_sumo']):
                prefx = str(ga)+'_'+str(counter)+'_'
                loopDataNameK = (prefx + 'out.xml')
                Loopdata_csv = self.paths["cache"] + prefx+'loopDataName.csv'
                Data2csv = ('python ' + self.paths["sumo"] + 'tools/xml/xml2csv.py ' + 
                            self.paths["cache"] + loopDataNameK + 
                            ' -o ' + Loopdata_csv)
                subprocess.run(Data2csv, shell=True)
                simulated_tripsInTable= pd.read_csv(Loopdata_csv, sep=";", header=0)
                all_counts = pd.DataFrame()
                for clock in np.arange(start, end, agg_inter):
                    startSimTime = clock*60*60 + fract*60
                    endSimTime = (clock+agg_inter)*60*60 + fract*60
                    interval = (simulated_tripsInTable['interval_begin']>=startSimTime)&(simulated_tripsInTable['interval_end'] <= endSimTime)
                    simulated_trips = simulated_tripsInTable.loc[interval,:].copy()
                    simulated_trips.loc[:,'EdgeID'] = simulated_trips.apply(lambda x: x['interval_id'].split('_')[1], axis=1)
                    simulated_trips = simulated_trips[['EdgeID', 'interval_entered']]
                    simulated_trips.columns = ['EdgeID', 'Counts_'+str(clock)+'_'+str(clock+agg_inter)]
                    grouped = simulated_trips.groupby('EdgeID').agg(np.sum)
                    all_counts = pd.concat([all_counts, grouped], axis=1)
                counts = pd.concat([counts, all_counts], axis=1)
            measurements_agg = counts.groupby(by=counts.columns, axis=1).apply(lambda f: f.mean(axis=1))
        # travel time
        if self.sumo_var['objective'] == 'time':
            time = pd.DataFrame()
            for counter in range(self.sumo_var['n_sumo']):
                prefx = str(ga)+'_'+str(counter)+'_'
                route_xml = self.paths["cache"] + prefx +'network.vehroutes.xml'
                route_csv = self.paths["cache"] + prefx +'network.vehroutes.csv'
                Data2csv = ('python ' + self.paths["sumo"] + 'tools/xml/xml2csv.py ' + 
                            route_xml + ' -x ' + self.paths["sumo"] + 'data/xsd/routes_file.xsd -o ' + route_csv)
                subprocess.run(Data2csv, shell=True)
                TT_output = pd.read_csv(route_csv, sep=';')
                TT_output = TT_output.loc[:,['vehicle_arrival', 'vehicle_depart', 'vehicle_fromTaz','vehicle_toTaz']]
                TT_output["Travel_time"] = (TT_output["vehicle_arrival"] - TT_output["vehicle_depart"])
                all_tt = pd.DataFrame()
                for clock in np.arange(start, end, agg_inter):
                    startSimTime = clock*60*60 + fract*60
                    endSimTime = (clock+agg_inter)*60*60 + fract*60
                    interval = (TT_output['vehicle_arrival']>=startSimTime)&(TT_output['vehicle_depart'] <= endSimTime)
                    simulated_tt = TT_output[interval]
                    simulated_tt = simulated_tt.loc[:, ['vehicle_fromTaz','vehicle_toTaz', 'Travel_time']]
                    grouped = simulated_tt.groupby(['vehicle_fromTaz','vehicle_toTaz']).mean()
                    grouped.columns = ['tt_'+str(clock)+'_'+str(clock+agg_inter)]
                    all_tt = pd.concat([all_tt, grouped], axis=1)
                time = pd.concat([time, all_tt], axis=1)
            measurements_agg = time.groupby(by=time.columns, axis=1).apply(lambda f: f.mean(axis=1))
        return measurements_agg
