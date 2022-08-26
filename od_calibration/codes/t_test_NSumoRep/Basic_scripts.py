# -*- coding: utf-8 -*-


import numpy as np
import math
import subprocess
#from pathos.multiprocessing import ProcessingPool as Pool
import re
import pandas as pd
from shutil import copyfile
from shutil import move
import os
import os.path

# =============================================================================
# A simple function that converts an OD matrix to an OD vector. We read
# through the matrix (row after row and column after column) and create a vector.
# =============================================================================

def Full2Vector(Full):
    k=0
    Vector = np.zeros((Full.shape[0]*Full.shape[1],1))
    for i in range(0, Full.shape[0]):
        for j in range(0, Full.shape[1]):
            Vector[k,0]=Full[i,j]
            k=k+1
    return Vector

# =============================================================================   
# #% A simple function that converts an OD vector to OD matrix. We read
# #% through the vector (row after row) and create a matrix.
# =============================================================================
    
def Vector2Full(allODArray):
    nZones = int(np.sqrt(allODArray.shape[0]))
    FullOD = np.zeros((nZones,nZones))
    m = 0
    for i in range(0, nZones):
        for j in range(0, nZones):
            FullOD[i,j]= allODArray[m];
            m=m+1;
    return FullOD


## =============================================================================
## '''% A function to read a SUMO based OD file and convert it to a readable and
## % usable OD in matlab. This includes the creation of "zones' masks", which
## % refers to a mapping between string-based representations of zones' IDs
## % and the corresponding indexes they receive in MATLAB. For example zone
## % with an ID of $@#35%$# will be code as "1" and there will be a row in
## % the Zones_masks where the initial ID and the coded ID will be in the same
## % time. '''
## =============================================================================

def OD_formatting(inputOD):

    TheOD = pd.read_csv(inputOD, sep = ' ', skiprows =5, header=None)
    if (math.sqrt(len(TheOD))-int(math.sqrt(len(TheOD)))):
        print("\nOD matrix is not a square matrix\n")
        FullOD=[]
        allODArray=[]
        return
    TheOD = np.transpose(np.array([TheOD[0],TheOD[1],TheOD[2]]))
    allODArray = TheOD[:,2]
    FullOD = Vector2Full(allODArray)
    Zones=list(set(TheOD[:,0]))
    Zones_masks = np.transpose(np.vstack((range(len(Zones)),Zones)))
    return FullOD, allODArray, Zones_masks



## =============================================================================
## ''' Reads a detector file and produces some basic representations of the
##  counts including the number of links counted and their IDs. Attention
##  should be placed on the ID section. '''
## =============================================================================

def readE1Counters(additionalName, pathToSUMOtools,pathToNetwork_a,OS):

    Detectors_csv=(r''+ pathToNetwork_a + '/' 'detectors.csv')
    Data2csv = (r'python ' '"' + pathToSUMOtools + 'tools/xml/xml2csv.py' '"'\
                ' ' '"' + pathToNetwork_a + '/' + additionalName + '"' ' -o '\
                '"' + Detectors_csv + '"')
    os.system(Data2csv)
    
    Detectors= pd.read_csv(Detectors_csv, sep=";", header=0)
    Detector_ID = Detectors['e1Detector_id']
    myEdges = [word.split('_')[1] for word in Detector_ID]
    e1counters = set(myEdges)
    nE1counters = len(e1counters)

    return  e1counters, nE1counters



## =============================================================================
## '''% Read the Loop Detectors Data: Each SUMO run produces a file with the
## % traffic counts. This function reads the corresponding traffic counts file
## % and converts the readings to traffic counts usable in Matlab. The process
## % is the simple xml2csv and readcsv process commonly used. '''
## I need to improve the code later by using better functions and reducing the
## code lines.    
## =============================================================================
#
def readstateLoopData(pathToSUMOtools,pathtoCaseOutput,loopDataName,endtime,OS):

    # We first do some time manipulations necessary.
    fract=float(endtime)
    integ=int(fract)
    fract=round(fract-integ, 2)
    endSimTime=integ*60*60 + fract*60
        
    Loopdata_csv=(r''+ pathtoCaseOutput + '/' 'loopDataName.csv')

    Data2csv = (r'python ' '"' + pathToSUMOtools + 'tools/xml/xml2csv.py' '"'\
                ' ' '"' + pathtoCaseOutput + '/' + loopDataName + '"' ' -o '\
                '"' + Loopdata_csv + '"')
    subprocess.run(Data2csv)

    # Then we read the data from this run (taking into account the proper time
    # interval).
    
    simulated_tripsInTable= pd.read_csv(Loopdata_csv, sep=";", header=0)
    simulated_tripsInTable = simulated_tripsInTable[simulated_tripsInTable['interval_end'] \
                                                    < endSimTime]

    # The final step is to aggregate the counts per link (so that we do can
    # estimate per link flow and not per lane). 
    Detector_ID=simulated_tripsInTable['interval_id']
    myEdges = [word.split('_')[1] for word in Detector_ID]
    simulated_tripsInTable['EdgeID'] = myEdges
    
    temp = pd.DataFrame()
    temp['EdgeID'] = myEdges
    temp['Counts'] = simulated_tripsInTable['interval_entered']
    Grouped = temp.groupby('EdgeID').agg(np.sum)
    Grouped['Edge'] = Grouped.index
    AllPeriodEdgesFlows = pd.DataFrame()
    AllPeriodEdgesFlows['Counts'] = Grouped['Counts']
    AllPeriodEdgesFlows = AllPeriodEdgesFlows.reset_index()
    AllPeriodEdgesFlows = AllPeriodEdgesFlows.values
    del temp, Grouped
    return AllPeriodEdgesFlows
    
## =============================================================================
## '''This function provides some measures of Goodness of Fit.'''
## =============================================================================
    
def gof_eval(truedata, simulatedCounts):
    truedata = truedata.reshape(len(truedata), 1)
    simulatedCounts = simulatedCounts.reshape(len(simulatedCounts),1)
    kk=truedata.shape
    RMSNE=[]
 #   GEH1= []
 #   MANE=[]
 #   MNE=[]
 #   U1=[]
 #   U2=[]
 #   U = []
 #   ME = []
 #   NME = [] 
 #   NRMSE = []
 #    NMAE = []
    for i in range(0, kk[1]):
     #   RMSE = RMSE.append(np.sqrt(sum((simulatedCounts[:,i]-truedata[:,i])**2)/kk[0]))
     #   SE = SE.append(sum(simulatedCounts[:,i]-truedata[:,i])**2)
     #   MAE = MAE.append(sum(abs(simulatedCounts[:,i]-truedata[:,i]))/kk[0])
     #   U1 = U1.append(np.sqrt(sum((simulatedCounts[:,i])**2)/kk[0]))
     #   U2 = U2.append(np.sqrt(sum((truedata[:,i])**2)/kk[0]))
     #   U = U.append(RMSE[i]/(U1[i]+U2[i]))
     #   ME = ME.append(sum((simulatedCounts[:,i]-truedata[:,i]))/(kk[0]))
     #   NME = NME.append((sum(simulatedCounts[:,i]-truedata[:,i]))/(sum(truedata[:,i])))
     #   NRMSE = NRMSE.append(np.sqrt(sum((simulatedCounts[:,i]-truedata[:,i])**2)*kk[0])/(sum(truedata[:,i])))
     #   NMAE = sum(abs(simulatedCounts[:,i]-truedata[:,i]))/(sum(abs(truedata[:,i])))
        RMSNE.append(0)
     #   MANE.append(0)
     #   MNE.append(0)
     #   GEH1.append(0)
        for j in range(0, kk[0]):
            if truedata[j,i]>0:
                RMSNE[i]=RMSNE[i]+((simulatedCounts[j,i]-truedata[j,i])/truedata[j,i])**2
     #           MANE[i]=MANE[i]+abs((simulatedCounts[j,i]-truedata[j,i])/truedata[j,i])
     #           MNE[i]=MNE[i]+(simulatedCounts[j,i]-truedata[j,i])/truedata[j,i]
            elif simulatedCounts[j,i]>0:
                RMSNE[i]=RMSNE[i]+1
                
     #           MANE[i]=MANE[i]+1
     #           MNE[i]=MNE[i]+1
     #       if truedata[j,i]>0 or simulatedCounts[j,i]>0:
     #           GEH=np.sqrt(2*(simulatedCounts[j,i]-truedata[j,i])^2/(simulatedCounts[j,i]+truedata[j,i]))
     #       if GEH<=1:
     #           GEH1[i]=GEH1[i]+1
     #       else:
     #           GEH1[i]=GEH1[i]+1
     #   MNE[i]=MNE[i]/kk[1]
        RMSNE[i]=np.sqrt(RMSNE[i]/kk[0])
     #   MANE[i]=MANE[i]/kk[0]
     #   GEH1[i]=(kk[1]-GEH1[i])/kk[0]
        #Here it is possible to set which GoF will be part of our analysis
        #y=[RMSE;RMSNE;NRMSE;GEH1;MAE;MANE;NMAE;SE;U;ME;MNE;NME];
    y = RMSNE    
    return y #, counts

## =============================================================================
## #% An introductory function that copies some important files in the temporal
## #% folder. 
## =============================================================================
#
def scenarioImport(ScenarioDataFolder, scenarioID, tempLocalLocation,\
                   examinedState,formatedNetworkStr):
    
    ScenarioDataFolder      = re.sub("\\\\","/", ScenarioDataFolder)
    #newStrTempLoc = tempLocalLocation.replace("\\","")   
    #tempLoc = newStrTempLoc + scenarioID

    #% Copy IMPORTANT FILES.

    ScenarioFiles = ScenarioDataFolder + scenarioID

    addOldLoc  = ScenarioFiles + "/" + "adTrCounts.xml"
    addNewLoc  = formatedNetworkStr + "/" + "adTrCounts.xml"
    copyfile(addOldLoc, addNewLoc)

    addOldLoc  = ScenarioFiles + "/" + "AVERAGE_" + str(examinedState) + "_" + "myODUpdated.txt"
    addNewLoc  = formatedNetworkStr + "/" + "AVERAGE_" + str(examinedState) + "_" + "myODUpdated.txt"
    copyfile(addOldLoc, addNewLoc)

    #% Copy True States
    addOldLoc  = ScenarioFiles + "/" + "TRUE_" + str(examinedState-1) + "_" + "state.xml"
    addNewLoc  = tempLocalLocation + scenarioID + "/" + str(examinedState-1) + "state.xml"
    copyfile(addOldLoc, addNewLoc)
#    
# 