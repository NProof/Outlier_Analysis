# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

# 合併目錄下全部的資料

def CombineXLSX(dir_spe):
    df = pd.DataFrame()
    
    files = dir_spe.iterdir()
    for file in files:
        print(file.name)
        if file.suffix == ".xlsx":
            df = df.append(pd.read_excel(file), ignore_index=True)
    return df.dropna()

# 將 [pd.Series] time_str 的 型別(dtype) 轉成 [datetime64[ns]]]
# 使用 to_datetime 的參數 errors，以此區別相異的格式
# P.S. -- str2Datetime(*).name is "timestamp"

def str2Datetime(str_series):
    # 必要的 (轉換格式) 前置作業
    str_series = str_series.apply(lambda _: str(_).replace('上午', 'AM').replace('下午', 'PM'))
    
    timestamp_series = pd.Series( [pd.NaT] * len(str_series), name = 'timestamp')
    mask = pd.Series( [True] * len(str_series) )
    
    for fmt in formats:
        timestamp_series[mask] = pd.to_datetime(str_series, format = fmt, errors='coerce')
        mask = timestamp_series.isnull()
    # 確保全部格式轉換成功
    assert(~mask.any())
    return timestamp_series

# 為了方便管理，每個月的資料集中於特定目錄 (origin_dir_path)
origin_dir_path = Path("../dataset/origin_version")
combine_file_path = Path("../dataset/M01-12.csv")

# 資料中 [時間] 所有的格式
formats = ["%Y/%m/%d %p %I:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/ %m / %d %p %I:%M:%S"]

# (Q1) 資料中的時間格式不一致
    # (A1) str2Datetime(str_series)

# step1 : Combine all of data into one file.
# step2 : ...

if __name__ == "__main__":
    # 確認目錄是否合法
    assert(origin_dir_path.exists())
    
    df_step1 = CombineXLSX(origin_dir_path)
    backup_df_step1 = df_step1.copy() # [BACKUP]
    time_str = df_step1.pop("時間")
    temps_step1 = str2Datetime(time_str)
    
    # 在儲存 df_step1 (.csv file) 之前
    # 把 index <- temps_step1 / columns name 去除 "\n"
    df_step1.index = temps_step1
    df_step1.columns = df_step1.columns.to_series().apply(lambda _ : _.replace("\n", ""))
    df_step1.to_csv(combine_file_path)
    
    del origin_dir_path, formats, temps_step1
    
    df_merge = pd.read_csv(combine_file_path)
    