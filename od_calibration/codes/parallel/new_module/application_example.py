# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:45:18 2022

@author: lasts
"""
import pandas as pd
import numpy as np
from preparation import DataPreparation
from calibration_algorithms import PC_SPSA

paths = dict(network='../networks/',
             sumo='D:/Sumo/',
             od_prior='../networks/demand/',
             measurements='../networks/measurements/',
             cache='../cache/'
             )

paras = dict(n_gen=1,
             a=1,
             c=0.15,
             A=25,
             alpha=0.3,
             gamma=0.15
             )
             # n_iter=5,
             # hist_method=6)

sumo_var = dict(network='network.net.xml',
                tazname='taZes.taz.xml',
                add_file='addition.add.xml',
                starttime='05:00',
                endtime='07:00',
                n_sumo=2,
                objective='counts',
                interval=1,
                w=0.1
                )

dp = DataPreparation(paths, sumo_var)
dp.load_path()
data, od_prior = dp.load_data()

calibrate = PC_SPSA(paths, sumo_var)
