# -*- coding: utf-8 -*-
"""
This file shows an example of defining `paths`, `paras`, `sumo_var`, and an example of applying this tool to calibrate a network.

@author: Qing-Long Lu (qinglong.lu@tum.de)
"""
import pandas as pd
import numpy as np
from preparation import DataPreparation
from calibration_algorithms import PC_SPSA

paths = dict(network='../networks/',
             sumo='D:/Sumo/',
             demand='../networks/demand/',
             measurements='../networks/measurements/',
             cache='../cache/'
             )

paras = dict(n_gen=1,
             a=1,
             c=0.15,
             A=25,
             alpha=0.3,
             gamma=0.15,
             variance=70,
             )
             # n_iter=5,
             # hist_method=6)

sumo_var = dict(network='network.net.xml',
                tazname='taZes.taz.xml',
                add_file='addition.add.xml',
                starttime='05.00',
                endtime='07.00',
                n_sumo=1,
                objective='counts',
                interval=1,
                w=0.1
                )

# prepare paths and data
dp = DataPreparation(paths, sumo_var)
dp.load_path()
data_true, od_prior = dp.load_data()

calibrate = PC_SPSA(paths, sumo_var, paras, od_prior, data_true)
# artificially generate a historical OD matrix, not needed if you already have a historical OD
hist_od = calibrate.create_history(hist_method=6, R=[0.3,0.4,1])

if __name__ == '__main__':
    # convergence, best_metrics, best_data, best_od
    result = calibrate.run(hist_od, n_iter=2)























