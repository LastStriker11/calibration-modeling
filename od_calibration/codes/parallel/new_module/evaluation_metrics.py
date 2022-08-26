# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:34:02 2022

@author: lasts
"""
import numpy as np

def normalized_root_mean_square(data_true, data_simulated, od_prior, od_calibrated, w):
    '''
    Normalized root mean square error.

    Parameters
    ----------
    data_true : DataFrame
        Observed traffic measurements.
    data_simulated : DataFrame
        Traffic measurements from simulations.
    od_true : DataFrame
        Prior OD estimates.
    od_calibrated : DataFrame
        OD estimates from current calibration evaluation.
    w : float
        weights of the errors imposed by OD matrix.

    Returns
    -------
    rmsn : list
        List of NRMSE values, one per interval.

    '''
    for i in range(data_simulated.shape[1]-1):
        data_simulated.iloc[:,i] = data_simulated.iloc[:,i]+data_simulated.iloc[:,i+1]
    data_true.columns = data_simulated.columns
    data_simulated = data_simulated.loc[data_true.index,:]
    diff = (data_true - data_simulated)**2
    n = diff.count()
    sum_diff = diff.sum()
    sum_true = data_true.sum()
    rmsn_count = np.sqrt(n*sum_diff)/sum_true
    
    # calculate the rmsn for ods
    od_prior = od_prior.iloc[:, 2:]
    od_calibrated = od_calibrated.iloc[:, 2:]
    diff_od = (od_prior - od_calibrated)**2
    n_od = diff_od.count()
    sum_diff_od = diff_od.sum()
    sum_true_od = od_prior.sum()
    rmsn_od = np.sqrt(n_od*sum_diff_od)/sum_true_od
    
    rmsn = (1-w)*rmsn_count.values + w*rmsn_od.values
    rmsn = rmsn.tolist()
    return rmsn