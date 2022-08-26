# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:32:31 2020

@author: moeid
"""

# -*- coding: utf-8 -*-
"""
Attention!
Counts Calibration: use line 47 and 48
Travel Time Calibration: use line 49 and 50
"""
import pandas as pd
import numpy as np
import shutil
import os
#%%
def load_path(network, scenario=False, objective='counts', method=0, NSumoRep=2, G=1, N=2, x=None, y=None):
    '''
    create dictionaries for paths which will be used in the following program.
    
    Parameters
    ----------
    network : string
        network of the study area.
    scenario : bool, optional
        if generate a new scenario. The default is False.
    objective : string, optional
        traffic measurement to use. The default is 'counts'.
    method : int, optional
        method used to generate the historical OD dataset. The default is ''.
    NSumoRep : int, optional
        number of SUMO simulation replications. The default is 2.
    G : int, optional
        number of generations for evaluating the algorithm. The default is 1.
    N : int, optional
        maximum iteration. The default is 2.
    x : float, optional
        demand reduction of the scenario. Set as None when scenario is True. The default is None.
    y : float, optional
        demand randomness of the scenario. Set as None when scenario is True. The default is None.

    Returns
    -------
    Paths : dictionary
        Contains the paths to the network, SUMO and cache.
    Network_var : dictionary
        Contains the names of necessary files and simulation setups.
    '''
    
    Paths = dict(pathToNetwork_a      = r'<path to the network folder>/',
                 pathToSUMO           = r'<path to sumo folder>/',
                 tempLocalLocation    = r'<path to cache folder>/')
            
    Network_var = dict(mesoNet          = 'network.net.xml',
                       tazname          = 'taZes.taz.xml',
                       loopDataName     = 'out.xml',
                       additionalName   = 'addition.add.xml',
                       starttime        = "05.00",
                       endtime          = "10.00",
                       NSumoRep         = NSumoRep,
                       trip_based       = True,
                       objective        = objective,
                       interval         = 1,
                       scenarioID       = network+'/',
                       w                = 0)
    
    Network_var['flag'] = (scenario)|(x is None)
    scenario_type = 'spatial_temporal'
    if (scenario == True) or (x is None):
        folder = 'Demand'
        Network_var['inputOD'] = [f for f in os.listdir(Paths['pathToNetwork_a']+folder) if f.endswith('.txt')]  
        Network_var['inputOD'].sort()
        
    if (scenario == False) & (x is not None):
        Network_var['para'] = 'scenarios/'+scenario_type+'/'+str(x)+'_'+str(y)
        folder = 'D_MandStartOD/'+str(method)
        Network_var['inputOD'] = [f for f in os.listdir(Paths['pathToNetwork_a']+folder) if 'start_od' in f]
        Network_var['inputOD'].sort()
        if Network_var['objective'] == 'counts':
            Network_var['truedata'] = os.listdir(Paths['pathToNetwork_a']+Network_var['para']+'/counts')
        if Network_var['objective'] == 'time':
            Network_var['truedata'] = os.listdir(Paths['pathToNetwork_a']+Network_var['para']+'/time')
        Network_var['truedata'].sort()
    
    # create the folder to store the scenario files
    Paths["pathToScenario"] = Paths["tempLocalLocation"]+Network_var["scenarioID"]
    if os.path.exists(Paths["pathToScenario"]):
        shutil.rmtree(Paths["pathToScenario"])
        os.mkdir(Paths["pathToScenario"])
    if not os.path.exists(Paths["pathToScenario"]):
        os.mkdir(Paths["pathToScenario"])
    Paths["pathToScenario"] = Paths["pathToScenario"]+str(method)+'/'
    if os.path.exists(Paths["pathToScenario"]):
        shutil.rmtree(Paths["pathToScenario"])
        os.mkdir(Paths["pathToScenario"])
    if not os.path.exists(Paths["pathToScenario"]):
        os.mkdir(Paths["pathToScenario"])
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
    
    '''
     copy relevant files for SUMO simulation to a folder under the temp directory,
     so that all output files will be located in a same folder which won't make the
     original network folder dirty and give us convenienc to deal with the temporary
     outputs which could be redundant.
    '''
    # mesoNet
    old_loc = Paths['pathToNetwork_a']+Network_var['mesoNet']
    new_loc = Paths['pathToScenario']+Network_var['mesoNet']
    shutil.copyfile(old_loc, new_loc)
    # taz
    old_loc = Paths['pathToNetwork_a']+Network_var['tazname']
    new_loc = Paths['pathToScenario']+Network_var['tazname']
    shutil.copyfile(old_loc, new_loc)
    # additional
    old_loc = Paths['pathToNetwork_a']+Network_var['additionalName']
    new_loc = Paths['pathToScenario']+Network_var['additionalName']
    shutil.copyfile(old_loc, new_loc)
    # inputOD
    for i in range(len(Network_var["inputOD"])):    
        old_loc = Paths['pathToNetwork_a']+folder+'/'+Network_var['inputOD'][i]
        new_loc = Paths['pathToScenario']+Network_var['inputOD'][i]
        shutil.copyfile(old_loc, new_loc)
        if Network_var['flag'] == False:
            # truedata
            old_loc = Paths['pathToNetwork_a']+Network_var['para']+'/'+Network_var['objective']+'/'+Network_var['truedata'][i]
            new_loc = Paths['pathToScenario']+Network_var['truedata'][i]
            shutil.copyfile(old_loc, new_loc)            
    # change the network folder to the temporal position
    Paths['origin_network'] = Paths['pathToNetwork_a']
    Paths['pathToNetwork_a'] = Paths['pathToScenario']
    return Paths, Network_var
#%%
'''
 import the data that will be used in the following program
'''
def load_data(Paths, Network_var):
    start = int(float(Network_var["starttime"]))
    end = int(float(Network_var['endtime']))
    cols = list(range(start, end))
    cols_inter = list(np.arange(start, end, Network_var['interval']))
    data = pd.DataFrame()
    if Network_var['flag'] == False:
        #===================== for counts
        if Network_var['objective'] == 'counts':
            for i in range(len(Network_var["truedata"])):
                temp_data = pd.read_csv(Paths["pathToNetwork_a"] +Network_var['truedata'][i], header=None)
                temp_data.set_index(0, inplace=True)
                data = pd.concat([data, temp_data], axis=1)
            data.columns = cols_inter
            #@@@@
            data = data[(data.T!=0).any()]
            #@@@@
        #==================================
        #=============== for travel time
        if Network_var['objective'] == 'time':
            for i in range(len(Network_var["truedata"])):
                temp_data = pd.read_csv(Paths["pathToNetwork_a"] +Network_var['truedata'][i], header=None)
                temp_data['label'] = list(zip(temp_data.iloc[:,0], temp_data.iloc[:,1]))
                temp_data.drop([0,1], axis=1, inplace=True)
                temp_data.set_index('label', inplace=True)
                data = pd.concat([data, temp_data], axis=1)
            data.columns = cols
        #===============================
        input_od = pd.DataFrame()
        for i in range(len(Network_var["inputOD"])):
            temp_od = pd.read_csv(Paths["pathToNetwork_a"] +  Network_var["inputOD"][i], sep='\s+', header=None, skiprows=5)
            input_od = pd.concat([input_od, temp_od.iloc[:,2]], axis=1)
        input_od.columns = cols_inter
        input_od = pd.concat([temp_od.iloc[:,:2], input_od], axis=1)
    elif Network_var['flag'] == True:
        data = None
        #========================
        input_od = pd.DataFrame()
        for i in range(len(Network_var["inputOD"])):
            temp_od = pd.read_csv(Paths["pathToNetwork_a"] +  Network_var["inputOD"][i], sep='\s+', header=None, skiprows=8)
            input_od = pd.concat([input_od, temp_od.iloc[:,2]], axis=1)
        input_od.columns = cols
        input_od = pd.concat([temp_od.iloc[:,:2], input_od], axis=1)
    return data, input_od
#%%
'''
 1. create the OD file which will be imported into SUMO for simulation
 2. create lists for the variables which will be used in the 'inner' parallel processing
    which means the parallel processing for SUMO simulation (SPSA_para['NSumoRep'])
'''
def write_od(Paths, Network_var, VectorODExamined, ga):
    start = int(float(Network_var["starttime"]))
    end = int(float(Network_var['endtime']))
    cols = list(np.arange(start, end, Network_var['interval']))
    cols = [0, 1] + cols
    VectorODExamined.columns = cols
    for i in range(VectorODExamined.shape[1]-2):
        tt    = start+Network_var['interval']*i
        inter = int(tt)
        fract = (tt - inter)*60
        if fract == 0:
            TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.00 ' 
                           + str(inter)+'.' + str(int(fract+60*Network_var['interval'])) + '\n* Factor \n1.00\n')
        if fract == 45:
            TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.' + str(45)+' ' 
                           + str(inter+1)+ '.00' + '\n* Factor \n1.00\n')
        else:
            TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.' + str(int(fract))+' ' 
                           + str(inter)+'.' + str(int(fract+60*Network_var['interval'])) + '\n* Factor \n1.00\n')
        ma = Paths["pathToScenario"] + 'OD_updated_'+str(VectorODExamined.columns[i+2])+'.txt'
        file = open(ma, 'w')
        file.write(TheBeginText)
        file.close()
        VectorODExamined.iloc[:, [0,1,i+2]].to_csv(ma, header=False, index=False, sep=' ', mode='a')
    # replicate the variables
    seedNN_vector = np.random.normal(0, 10000, Network_var["NSumoRep"]).tolist()
    PATH = []
    NETWORK = []
    COUNTER = list(range(0,Network_var['NSumoRep']))
    GENERATION =[]
    for num in range(Network_var['NSumoRep']):
        PATH.append(Paths)
        NETWORK.append(Network_var)
        GENERATION.append(ga)
    return PATH, NETWORK, COUNTER, seedNN_vector, GENERATION
#%%
'''
 commands for SUMO simulation, i.e. generate upd_ODTrips.trips.xml, and then 
 generate out.xml, vehroutes.xml
'''
def RSUMOStateInPar(Paths, Network_var, k, seedNN, ga):
    import subprocess
    start = int(float(Network_var["starttime"]))
    end = int(float(Network_var['endtime']))
    rerouteProb = 0.1
    cols = list(np.arange(start, end, Network_var['interval']))
    # OD2TRIPS, generate upd_ODTrips.trips.xml
    prefx = str(ga)+'_'+str(k)+'_'
    seedNN = int(seedNN)
    oD2TRIPS = '"'+ Paths["pathtoSUMO"]+'bin/od2trips" --no-step-log --output-prefix '+\
        prefx+ ' --spread.uniform --taz-files '+ '"'+ Paths["pathToNetwork_a"] + Network_var["tazname"] +'"'+' -d '
        
    for i in range(len(cols)):
        oD2TRIPS = oD2TRIPS + '"' + Paths["pathToScenario"] + 'OD_updated_'+str(cols[i])+'.txt' + '",'
    oD2TRIPS = oD2TRIPS[:-1]
    oD2TRIPS = oD2TRIPS + ' -o '+'"'+ Paths["pathToScenario"]+'upd_ODTrips.trips.xml'+ '"'+\
            ' --seed '+str(seedNN)
    '''
    If passing a single string, either shell must be True (see below) or else the string must simply 
    name the program to be executed without specifying any arguments.
    '''
    subprocess.run(oD2TRIPS, shell=True)
    
    # generate out.xml, vehroutes.xml
    sumoRUN = '"'+Paths["pathtoSUMO"]+'bin/sumo" --mesosim  --no-step-log -W --output-prefix '+\
        prefx+' -n '+'"'+Paths["pathToNetwork_a"]+Network_var["mesoNet"]+'"'+\
        ' -b '+Network_var['beginSimTime']+' -e '+ Network_var['endSimTime']+\
        ' -r '+'"'+Paths["pathToScenario"]+prefx+'upd_ODTrips.trips.xml'+'"'+\
        ' --vehroutes ' + '"'+ Paths["pathToScenario"] + 'Munich.vehroutes.xml'+'"'\
        ' --additional-files '+'"'+ Paths["pathToNetwork_a"] + Network_var["additionalName"] +'"'+\
        ' --xml-validation never'+' --device.rerouting.probability '+str(rerouteProb)+\
        ' --seed '+str(seedNN)
    subprocess.run(sumoRUN, shell=True)
#%%
'''
 average all out.xml(for counts) or vehroutes.xml(for travel time) which are from 
 different SUMO runs
'''
def SUMO_aver(Paths, Network_var, ga):
    import subprocess
    import numpy as np
    
    fract = float(Network_var["starttime"])
    start = int(fract)
    end = int(float(Network_var['endtime']))
    fract = round(fract-start, 2)
    agg_inter = Network_var['interval']
    #===================== for counts==============
    if Network_var['flag'] == False:
        if Network_var['objective'] == 'counts':
            counts = pd.DataFrame()
            for counter in range(Network_var['NSumoRep']):
                prefx = str(ga)+'_'+str(counter)+'_'
                loopDataNameK = (prefx + 'out.xml')
                Loopdata_csv=(r''+ Paths["pathToNetwork_a"] + prefx+'loopDataName.csv')
                Data2csv = (r'python ' '"' + Paths["pathToSUMO"] + 'tools/xml/xml2csv.py' '"'\
                            ' ' '"' + Paths["pathToNetwork_a"] + '/' + loopDataNameK + '"' ' -o '\
                            '"' + Loopdata_csv + '"')
                subprocess.run(Data2csv, shell=True)
                simulated_tripsInTable= pd.read_csv(Loopdata_csv, sep=";", header=0)
                all_counts = pd.DataFrame()
                for clock in np.arange(start, end, agg_inter):
                    startSimTime = clock*60*60 + fract*60
                    endSimTime = (clock+agg_inter)*60*60 + fract*60
                    interval = (simulated_tripsInTable['interval_begin']>=startSimTime)&(simulated_tripsInTable['interval_end'] <= endSimTime)
                    simulated_trips = simulated_tripsInTable[interval]
                    simulated_trips['EdgeID'] = [word.split('_')[1] for word in simulated_trips.loc[:,'interval_id']]
                    simulated_trips = simulated_trips[['EdgeID', 'interval_entered']]
                    simulated_trips.columns = ['EdgeID', 'Counts_'+str(clock)+'_'+str(clock+agg_inter)]
                    grouped = simulated_trips.groupby('EdgeID').agg(np.sum)
                    all_counts = pd.concat([all_counts, grouped], axis=1)
                counts = pd.concat([counts, all_counts], axis=1)
            SimulatedCounts = counts.groupby(by=counts.columns, axis=1).apply(lambda f: f.mean(axis=1))
            return SimulatedCounts
        #================================================
        #================== for travel time=============
        if Network_var['objective'] == 'time':
            time = pd.DataFrame()
            for counter in range(Network_var['NSumoRep']):
                prefx = str(ga)+'_'+str(counter)+'_'
                route_xml = Paths["pathToScenario"] + prefx +'Munich.vehroutes.xml'
                route_csv = Paths["pathToScenario"] + prefx +'Munich.vehroutes.csv'
                Data2csv = (r'python ' '"' + Paths["pathToSUMO"] + 'tools/xml/xml2csv.py' '"'\
                            ' ' '"' + route_xml + '"' + ' -x ' '"' + Paths["pathToSUMO"] + 'data/xsd/routes_file.xsd' '"' ' -o '\
                            '"' + route_csv + '"')
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
            SimulatedTime = time.groupby(by=time.columns, axis=1).apply(lambda f: f.mean(axis=1))
            return SimulatedTime
    elif Network_var['flag'] == True:
        counts = pd.DataFrame()
        for counter in range(Network_var['NSumoRep']):
            prefx = str(ga)+'_'+str(counter)+'_'
            loopDataNameK = (prefx + 'out.xml')
            Loopdata_csv=(r''+ Paths["pathToNetwork_a"] + prefx+'loopDataName.csv')
            Data2csv = (r'python ' '"' + Paths["pathToSUMO"] + 'tools/xml/xml2csv.py' '"'\
                        ' ' '"' + Paths["pathToNetwork_a"] + '/' + loopDataNameK + '"' ' -o '\
                        '"' + Loopdata_csv + '"')
            subprocess.run(Data2csv, shell=True)
            simulated_tripsInTable= pd.read_csv(Loopdata_csv, sep=";", header=0)
            all_counts = pd.DataFrame()
            for clock in np.arange(start, end, agg_inter):
                startSimTime = clock*60*60 + fract*60
                endSimTime = (clock+agg_inter)*60*60 + fract*60
                interval = (simulated_tripsInTable['interval_begin']>=startSimTime)&(simulated_tripsInTable['interval_end'] <= endSimTime)
                simulated_trips = simulated_tripsInTable[interval]
                simulated_trips['EdgeID'] = [word.split('_')[1] for word in simulated_trips.loc[:,'interval_id']]
                simulated_trips = simulated_trips[['EdgeID', 'interval_entered']]
                simulated_trips.columns = ['EdgeID', 'Counts_'+str(clock)+'_'+str(clock+agg_inter)]
                grouped = simulated_trips.groupby('EdgeID').agg(np.sum)
                all_counts = pd.concat([all_counts, grouped], axis=1)
            counts = pd.concat([counts, all_counts], axis=1)
        SimulatedCounts = counts.groupby(by=counts.columns, axis=1).apply(lambda f: f.mean(axis=1))
        #===============================
        time = pd.DataFrame()
        for counter in range(Network_var['NSumoRep']):
            prefx = str(ga)+'_'+str(counter)+'_'
            route_xml = Paths["pathToScenario"] + prefx +'Munich.vehroutes.xml'
            route_csv = Paths["pathToScenario"] + prefx +'Munich.vehroutes.csv'
            Data2csv = (r'python ' '"' + Paths["pathToSUMO"] + 'tools/xml/xml2csv.py' '"'\
                        ' ' '"' + route_xml + '"' + ' -x ' '"' + Paths["pathToSUMO"] + 'data/xsd/routes_file.xsd' '"' ' -o '\
                        '"' + route_csv + '"')
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
        SimulatedTime = time.groupby(by=time.columns, axis=1).apply(lambda f: f.mean(axis=1))
        return SimulatedCounts, SimulatedTime
    #=================================================
#%%
'''
 perform plus perturbation and minus perturbation
'''
def generation_PC_SPSA(ga, input_od, ck, data, n_compon, z, V, index_same, Paths, Network_var, tmap):
    import pandas as pd
    import numpy as np
    import os
    from Functions import write_od, RSUMOStateInPar, SUMO_aver, gof_eval_od
    start_sim = int(float(Network_var["starttime"]))
    # build different folders for different generation
    if not os.path.exists(Paths['pathToScenario']+str(ga)):
        os.makedirs(Paths['pathToScenario']+str(ga))
    Paths['pathToScenario'] = Paths['pathToScenario']+str(ga)+'/'
    
    Deltas = []
    OD_plus = pd.DataFrame()
    for i in range(len(Network_var["inputOD"])):
        
        delta = 2*np.random.binomial(n=1, p=0.5, size=n_compon[i])-1 # Bernoulli distribution
        # plus perturbation
        z_per = z[i] + z[i]*ck*delta
        temp_per = pd.DataFrame(np.matmul(V[i],z_per))
        temp_per = pd.concat([input_od.iloc[:,:2], temp_per], axis=1)
        temp_per.columns = ['from', 'to', 'counts']
        temp_per.loc[index_same.iloc[:,i], 'counts'] = input_od.loc[index_same.iloc[:,i], start_sim+i*Network_var['interval']]
        index_neg = temp_per.loc[:,'counts']<3
        temp_per.loc[index_neg, 'counts'] = input_od.loc[index_neg, start_sim+i*Network_var['interval']]
        OD_plus = pd.concat([OD_plus, temp_per['counts']], axis=1)
        Deltas.append(delta)
    OD_plus = pd.concat([input_od.iloc[:,:2], OD_plus], axis=1)
    print('Simulation %d . plus perturbation' % ga)
    PATH, NETWORK, COUNTER, seedNN_vector, GENERATION = write_od(Paths, Network_var, OD_plus, ga)
    tmap(RSUMOStateInPar, PATH, NETWORK, COUNTER, seedNN_vector, GENERATION)
    data_simulated = SUMO_aver(Paths, Network_var, ga)
#    data_simulated = data_simulated.dropna(axis=0)
    y_plus = gof_eval_od(data, data_simulated, input_od, OD_plus, Network_var['w'])
    print('Plus: ', y_plus)
    
    # minus perturbation
    OD_minus = pd.DataFrame()
    for i in range(len(Network_var["inputOD"])):
        delta = Deltas[i] # Bernoulli distribution
        # plus perturbation
        z_per = z[i] - z[i]*ck*delta
        temp_per = pd.DataFrame(np.matmul(V[i],z_per))
        temp_per = pd.concat([input_od.iloc[:,:2], temp_per], axis=1)
        temp_per.columns = ['from', 'to', 'counts']
        temp_per.loc[index_same.iloc[:,i], 'counts'] = input_od.loc[index_same.iloc[:,i], start_sim+i*Network_var['interval']]
        index_neg = temp_per.loc[:,'counts']<3
        temp_per.loc[index_neg, 'counts'] = input_od.loc[index_neg, start_sim+i*Network_var['interval']]
        OD_minus = pd.concat([OD_minus, temp_per['counts']], axis=1)
    OD_minus = pd.concat([input_od.iloc[:,:2], OD_minus], axis=1)
    print('Simulation %d .minus perturbation' % ga)
    PATH, NETWORK, COUNTER, seedNN_vector, GENERATION = write_od(Paths, Network_var, OD_minus, ga)
    tmap(RSUMOStateInPar, PATH, NETWORK, COUNTER, seedNN_vector, GENERATION)
    data_simulated = SUMO_aver(Paths, Network_var, ga)
#    data_simulated = data_simulated.dropna(axis=0)
    y_minus = gof_eval_od(data, data_simulated, input_od, OD_minus, Network_var['w'])
    print('Minus: ', y_minus)
    
    # Gradient Evaluation
    g_hat = pd.DataFrame()
    for i in range(len(Network_var["inputOD"])):
        temp_g = (y_plus[i] - y_minus[i])/(2*ck*Deltas[i])
        g_hat = pd.concat([g_hat, pd.DataFrame(temp_g)], axis=1)
    return g_hat
#%%
'''
 create lists for the variables which will be used in the 'outer' parallel processing
 which means the parallel processing for different generations (SPSA_para['G'])
'''
def rep_generation_PC_SPSA(G, input_od, ck, data, n_compon, z, V, index_same, Paths, Network_var):
    GA = list(range(0, G))
    INPUT_OD = []
    CK = []
    DATA = []
    N_COMPON = []
    ZS = []
    VS = []
    INDEX_SAME = []
    PATH = []
    NETWORK = []  
    for num in range(G):
        INPUT_OD.append(input_od)
        CK.append(ck)
        DATA.append(data)
        N_COMPON.append(n_compon)
        ZS.append(z)
        VS.append(V)
        INDEX_SAME.append(index_same)
        PATH.append(Paths)
        NETWORK.append(Network_var)
    return GA, INPUT_OD, CK, DATA, N_COMPON, ZS, VS, INDEX_SAME, PATH, NETWORK
#%%
def gof_eval_od(data, data_simulated, od_true, od_calibrated, w):
    for i in range(data_simulated.shape[1]-1):
        data_simulated.iloc[:,i] = data_simulated.iloc[:,i]+data_simulated.iloc[:,i+1]
    data.columns = data_simulated.columns
    data_simulated = data_simulated.loc[data.index,:]
    diff = (data - data_simulated)**2
    n = diff.count()
    sum_diff = diff.sum()
    sum_true = data.sum()
    rmsn_count = np.sqrt(n*sum_diff)/sum_true
    
    # calculate the rmsn for ods
    od_true = od_true.iloc[:, 2:]
    od_calibrated = od_calibrated.iloc[:, 2:]
    diff_od = (od_true - od_calibrated)**2
    n_od = diff_od.count()
    sum_diff_od = diff_od.sum()
    sum_true_od = od_true.sum()
    rmsn_od = np.sqrt(n_od*sum_diff_od)/sum_true_od
    
    rmsn = (1-w)*rmsn_count.values + w*rmsn_od.values
    rmsn = rmsn.tolist()
    return rmsn
#%%
def create_Dm(Paths, Network_var, input_od, method, R):
    hist_days = 100
    D_m = pd.DataFrame()
    np.random.seed(1)
    if not os.path.exists(Paths['origin_network'] + 'D_MandStartOD'):
        os.mkdir(Paths['origin_network'] + 'D_MandStartOD')
    if not os.path.exists(Paths['origin_network'] + 'D_MandStartOD/'+str(method)):
        os.mkdir(Paths['origin_network'] + 'D_MandStartOD/'+str(method))
    folder = 'D_MandStartOD/'+str(method)
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
            # create the starting od
            start_od = pd.concat([od.iloc[:,:2], temp_dm.iloc[:,-1].astype('int32')], axis=1)
            inter = int(input_od.columns[i+2])
            fract = (input_od.columns[i+2]-inter)*60
            if fract == 0:
                TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.00 ' 
                               + str(inter)+'.' + str(int(fract+60*Network_var['interval'])) + '\n* Factor \n1.00\n')
            if fract == 45:
                TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.' + str(45)+' ' 
                               + str(inter+1)+ '.00' + '\n* Factor \n1.00\n')
            else:
                TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.' + str(int(fract))+' ' 
                               + str(inter)+'.' + str(int(fract+60*Network_var['interval'])) + '\n* Factor \n1.00\n')
            ma = Paths["origin_network"] +folder +'/'+str(input_od.columns[i+2])+'start_od.txt'
            file = open(ma, 'w')
            file.write(TheBeginText)
            file.close()
            start_od.to_csv(ma, header=False, index=False, sep=' ', mode='a')
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
        for t in range(temp_dm.shape[1]):
            start_od = pd.concat([input_od.iloc[:,[0,1]], temp_dm.iloc[:,t].astype('int32')], axis=1)
            inter = int(input_od.columns[t+2])
            fract = (input_od.columns[t+2]-inter)*60
            if fract == 0:
                TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.00 ' 
                               + str(inter)+'.' + str(int(fract+60*Network_var['interval'])) + '\n* Factor \n1.00\n')
            if fract == 45:
                TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.' + str(45)+' ' 
                               + str(inter+1)+ '.00' + '\n* Factor \n1.00\n')
            else:
                TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.' + str(int(fract))+' ' 
                               + str(inter)+'.' + str(int(fract+60*Network_var['interval'])) + '\n* Factor \n1.00\n')
            ma = Paths["origin_network"] + folder +'/'+str(input_od.columns[t+2])+'start_od.txt'
            file = open(ma, 'w')
            file.write(TheBeginText)
            file.close()
            start_od.to_csv(ma, header=False, index=False, sep=' ', mode='a')

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
        for t in range(temp_dm.shape[1]):
            start_od = pd.concat([input_od.iloc[:,[0,1]], temp_dm.iloc[:,t].astype('int32')], axis=1)
            inter = int(input_od.columns[t+2])
            fract = (input_od.columns[t+2]-inter)*60
            if fract == 0:
                TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.00 ' 
                               + str(inter)+'.' + str(int(fract+60*Network_var['interval'])) + '\n* Factor \n1.00\n')
            if fract == 45:
                TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.' + str(45)+' ' 
                               + str(inter+1)+ '.00' + '\n* Factor \n1.00\n')
            else:
                TheBeginText= ('$OR;D2 \n* From-Time  To-Time \n' + str(inter) +'.' + str(int(fract))+' ' 
                               + str(inter)+'.' + str(int(fract+60*Network_var['interval'])) + '\n* Factor \n1.00\n')
            ma = Paths["origin_network"] + folder +'/'+str(input_od.columns[t+2])+'start_od.txt'
            file = open(ma, 'w')
            file.write(TheBeginText)
            file.close()
            start_od.to_csv(ma, header=False, index=False, sep=' ', mode='a')
            
    # if method == 7: #create days based correlation
    # if method == 8: # no correlation create uniform distribution random numbers
    return D_m
