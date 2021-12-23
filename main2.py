# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 14:29:13 2021

@author: User
"""

import pandas as pd

import data_processing

if __name__ == '__main__':
    
    history = pd.read_csv('./dataset/Merge.csv').dropna()
    # cut parition
    history = history.iloc[:25000]
    history = data_processing.deduplicate(history, subject='時間')
    # add timestamp
    t_sp_h = history['時間'].apply(lambda _ : pd.to_datetime(_))
    time_h = data_processing.convertTime(t_sp_h)
    history.index = t_sp_h
    
    # PART TEST WITH MODIFIED ERROR

    err = pd.read_csv('./dataset/err-0001.csv').dropna()
    # cut parition
    err = err.iloc[:25000]
    err = data_processing.deduplicate(err, subject='time')
    # add timestamp
    t_sp_e = history['時間'].apply(lambda _ : pd.to_datetime(_))
    time_e = data_processing.convertTime(t_sp_h)
    err.index = t_sp_e
    