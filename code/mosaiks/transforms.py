"""This module contains functions for transforming datasets"""

import numpy as np
import pandas as pd
import warnings

from mosaiks.utils.logging import log_text



def dropna_Y(Y):
    Y = Y.squeeze()
    
    # mark locations with missing labels:
    if type(Y[0]) != str: #for float/int
        valid = ~np.isnan(Y) & ~np.isinf(Y) & ~(Y == -999)
    else: #for string
        valid = ~pd.isnull(Y)

    # check all columns if there are more than one (this is used
    # if continent IDs are merged in for residualizing by contitent)
    if len(Y.shape) > 1 and Y.shape[1] > 1:
        valid = valid.all(axis=1)

    # drop
    Y = Y[valid]

    return Y, valid


def dropna(X, Y, locations):
    Y = Y.squeeze()
    
    # drop obs with missing labels:
    Y, valid = dropna_Y(Y)
    locations = locations[valid]
    X = X[valid]
    return X, Y, locations


def custom_log_transform(Y):
    # If 0s present (but no negatives), add 1st percentile to make sure we dont have zeros to log
    if all(y == 0 for y in Y):
        log_text('All elements equal to 0. Cannot do log transform',print_text='warn')
    elif min(Y) >0:
        # If no zeros we'll just take the log
        Y = np.log(Y)
        pre_transform_added_value = 0              
    elif min(Y) <= 0 :
        log_text("Negative values or 0s present, adding absolute value of minimum and 1st percentile before taking log")
        # If there are negatives add absolute values of ymin & y first percentile
        pre_transform_added_value = np.abs(Y.min()) # Shift so that 0 is min(Y). If min(Y)==0, no shift occurs.
        pre_transform_added_value += np.percentile(Y[Y > 0] ,1) #Get first percentile of nonzero obs 
        Y = np.log(Y + pre_transform_added_value)
        
    return Y, pre_transform_added_value

def custom_log_val(val,Y):
    
    if val is None:
        return None

    if np.min(Y) > 0:
        if np.min(val) > 0:
            return np.log(val)
        else:
            return None
    
    elif np.min(Y) == 0:
        val  = val + np.percentile(Y[Y!=0],1)
        val = np.log(val) 
        return val
    
    elif np.min(Y) < 0 :
        # If there are negatives add absolute values of ymin and y first percentile of positive observations
        first_percentile = np.percentile(Y[Y > 0] ,1)
        
        val = val + np.abs(Y.min())
        val = np.log(val + first_percentile)
        return val




##################################################

