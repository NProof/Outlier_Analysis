# -*- coding: utf-8 -*-
# 用於處理資料的函式庫

import pandas as pd

# 
def deduplicate(df, subject):
    return df[~df[subject].duplicated()]

def convertTime(timestamp):
    date = timestamp.apply(lambda _ : _.date())
    hour = timestamp.apply(lambda _ : _.hour)
    minute = timestamp.apply(lambda _ : _.minute)
    time = pd.DataFrame(data={'d':date.values, 'h':hour.values, 'm':minute.values}, index=timestamp)
    return time