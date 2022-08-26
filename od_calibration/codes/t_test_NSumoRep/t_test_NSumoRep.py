# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:28:30 2019

@author: User
"""
import pandas as pd
import numpy as np
import subprocess
import Basic_scripts as bs
#%% paths definition
Paths = dict(pathToNetwork_a      = r'E:\Google_Drive_QL\HiWi\PC-SPSA/Munich_new/data/',\
             python               = r'D:\Anaconda3/python.exe',\
             pathToSUMOtools      = r'D:\Sumo/',\
             pathtoScripts        = r'E:\Google_Drive_QL\HiWi\PC-SPSA\scenario_generator/',\
             pathtoSUMOBIN        = r'D:\Sumo/bin/',\
             tempLocalLocation    = r'E:\Google_Drive_QL\HiWi\PC-SPSA\Munich_new\temp/',\
             ScenarioDataFolder   = r'E:\Google_Drive_QL\HiWi\PC-SPSA/Munich_new/data/',\
             OS                   = 'windows')

Network_var = dict(mesoNet          = 'Munich_cc2_mr.net.xml',\
                   tazname          = 'taZes2.taz.xml',\
                   loopDataName     = 'out.xml',\
                   additionalName   = 'temp2.add.xml',\
                   inputOD          = 'start_od_congested.txt',\
                   tripInfo         = 'tripInfo.xml',\
                   starttime        = "08.00",\
                   endtime          = "09.00",\
                   NSumoRep         = 100)
Paths["pathtoCaseOutput"] = Paths["tempLocalLocation"]
#% Start Simulation Times
integ= np.floor(np.double(Network_var["starttime"]))
fract=100*(np.double(Network_var["starttime"])-integ)
beginSim = int(integ*60*60 + fract*60)
#% End Simulation Times
integ=np.floor(np.double(Network_var["endtime"]))
fract=100*(np.double(Network_var["endtime"])-integ)
endSim = int(integ*60*60 + fract*60)

Network_var['beginSimTime'] =  str(beginSim)
Network_var['endSimTime']   =  str(endSim+500)
#%% running function definition
def RSUMOStateInPar(Paths, Network_var, k, seedNN_vector):
    rerouteProb = 0.1
    seedNN = int(np.around(seedNN_vector[k-1]))
    oD2TRIPS = '"'+ Paths["pathtoSUMOBIN"]+'od2trips" --no-step-log --output-prefix '+\
        str(k)+ ' --spread.uniform --taz-files '+ '"'+ Paths["pathToNetwork_a"] + Network_var["tazname"] +'"'+' -d ' +\
    '"' + Paths["pathtoCaseOutput"]+'myODUpdated.txt'+ '"'+\
    ' -o '+'"'+ Paths["pathtoCaseOutput"]+'upd_ODTrips.trips.xml'+ '"'+\
    ' --seed '+str(seedNN)
    subprocess.run(oD2TRIPS)
            
    sumoRUN = '"'+Paths["pathtoSUMOBIN"]+'sumo" --mesosim  --no-step-log --output-prefix '+\
            str(k)+' -n '+'"'+Paths["pathToNetwork_a"]+Network_var["mesoNet"]+'"'+\
            ' -b '+Network_var['beginSimTime']+' -e '+ Network_var['endSimTime']+\
            ' -r '+'"'+Paths["pathtoCaseOutput"]+str(k)+'upd_ODTrips.trips.xml'+'"'+\
            ' --vehroutes ' + '"'+ Paths["pathtoCaseOutput"] + 'Munich.vehroutes.xml'+'"'\
            ' --additional-files '+'"'+ Paths["pathToNetwork_a"] + Network_var["additionalName"] +'"'+\
            ' --xml-validation never'+' --device.rerouting.probability '+str(rerouteProb)+\
            ' --seed '+str(seedNN)
            
    subprocess.run(sumoRUN)
#%% import OD
input_od = pd.read_csv(Paths["pathToNetwork_a"] + Network_var["inputOD"], sep='\s+', header=None, skiprows=5)
#%% create the setup file for SUMO
TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + Network_var["starttime"] +' ' + Network_var["endtime"] + '\n* Factor \n1.00\n')
ma = Paths["pathtoCaseOutput"] + 't_test_setup.txt'
file = open(ma, 'w')
file.write(TheBeginText)
file.close()
input_od.to_csv(ma, header=False, index=False, sep=' ', mode='a')
#%% replicate sumo running
seedNN_vector = np.random.normal(0, 10000, Network_var["NSumoRep"])
counts = pd.DataFrame()
time = pd.DataFrame()
for counter in range(Network_var['NSumoRep']):
    print('The %d simulation.' % counter)
    RSUMOStateInPar(Paths, Network_var, counter, seedNN_vector)
    
    route_xml = Paths["pathtoCaseOutput"] + str(counter) +'Munich.vehroutes.xml'
    route_csv = Paths["pathtoCaseOutput"] + str(counter) +'Munich.vehroutes.csv'
    Data2csv = (r'python ' '"' + Paths["pathToSUMOtools"] + 'tools/xml/xml2csv.py' '"'\
                ' ' '"' + route_xml + '"' + ' -x ' '"' + Paths["pathToSUMOtools"] + 'data/xsd/routes_file.xsd' '"' ' -o '\
                '"' + route_csv + '"')
    subprocess.run(Data2csv)
    # travel time
    TT_output = pd.read_csv(route_csv, sep=';')
    TT_output["Travel_time"] = (TT_output["vehicle_arrival"] - TT_output["vehicle_depart"])
    TT_outp = TT_output.groupby(['vehicle_fromTaz','vehicle_toTaz']).mean()
    time = pd.concat([time, TT_outp['Travel_time']], axis=1)
    # counts
    loopDataNameK = (str(counter) + 'out.xml')
    Readings = bs.readstateLoopData(Paths["pathToSUMOtools"],Paths["pathToNetwork_a"],loopDataNameK,Network_var["endtime"],Paths["OS"])
    Readings = pd.DataFrame(Readings)
    Readings.columns = ['label', 'counts']
    Readings.set_index('label', inplace=True)
    counts = pd.concat([counts, Readings], axis=1)

#%% result output
counts.to_csv('t_test_counts100.csv', header=False)
time.to_csv('t_test_time100.csv', header=False)