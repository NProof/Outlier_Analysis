# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

def step1():
    # 確認目錄是否合法
    assert(origin_dir_path.exists())
    
# [START] # 合併目錄下全部的資料

    # # If you rum main.py, first run above @
    # df_step1 = backup_df_step1.copy() # @------------------
    
    df_step1 = pd.DataFrame()
    
    files = origin_dir_path.iterdir()
    for file in files:
        print(file.name)
        if file.suffix == ".xlsx":
            df_step1 = df_step1.append(pd.read_excel(file), ignore_index=True)
         
    _ = df_step1[~df_step1["時間"].duplicated()]
    df_step1 = _.dropna()
    
# [END] #

    time_str = df_step1.pop("時間")
    
# [START] # 將 [pd.Series] time_str 的型別 (dtype) 轉成 [datetime64[ns]]]
    time_str_2 = time_str.apply(lambda _: str(_).replace('上午', 'AM').replace('下午', 'PM'))
    temps_step1 = pd.Series( [pd.NaT] * len(time_str), name = 'timestamp')
    mask = pd.Series( [True] * len(time_str) )
    
    formats = ["%Y/%m/%d %p %I:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/ %m / %d %p %I:%M:%S"]
    for fmt in formats:
        temps_step1[mask] = pd.to_datetime(time_str_2, format = fmt, errors='coerce')
        mask = temps_step1.isnull()
    # 確保全部格式轉換成功
    # str_Illegal = temps_step1[mask]
    assert(~mask.any())
# [END] #

    df_step1_2 = df_step1.copy()
    
    # 在儲存 df_step1 (.csv file) 之前
    # 把 index <- temps_step1 / columns name 去除 "\n"
    df_step1_2.index = temps_step1
    
    # [Hypothesis 1] 時間特徵可能重複嗎?
    # df_step1_2 = df_step1_2[~temps_step1.duplicated()]
    # df_step1_2.index = df_step1_2[~temps_step1.duplicated()]
    
    df_step1_2.columns = df_step1_2.columns.to_series().apply(lambda _ : _.replace("\n", ""))
    df_step1_2.to_csv(combine_file_path)
    
# 為了方便管理，每個月的資料集中於特定目錄 (origin_dir_path)
origin_dir_path = Path("../dataset/origin_version")
combine_file_path = Path("../dataset/M01-12.csv")

if __name__ == "__main__":
    step1()
    df_merge_step1 = pd.read_csv(combine_file_path)
    
# [Hypothesis 1] 時間特徵可能重複嗎?
