# -*- coding: utf-8 -*-
# 用於處理資料的函式庫

import pandas as pd
from feature_selection import PFA

# 
def deduplicate(df, subject):
    return df[~df[subject].duplicated()]

def convertTime(timestamp):
    date = timestamp.apply(lambda _ : _.date())
    hour = timestamp.apply(lambda _ : _.hour)
    minute = timestamp.apply(lambda _ : _.minute)
    time = pd.DataFrame(data={'d':date.values, 'h':hour.values, 'm':minute.values}, index=timestamp)
    return time

def featureSelecting(f_df_raw):
    X = f_df_raw
    pfa = PFA(n_features=4)
    pfa.fit(X)
    
    # To get the transformed matrix
    X = pfa.features_
    
    # To get the column indices of the kept features
    column_indices = pfa.indices_
    
    column_names = f_df_raw.columns[column_indices]
    if '冰機Q(sum)' not in column_names:
        column_names = column_names.insert(0, '冰機Q(sum)')
    return column_names