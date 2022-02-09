# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import uniform

def dataModify_v1(df):
    noise = np.random.multivariate_normal(df.mean(), df.cov(), df.shape[0])
    
    r = uniform.rvs(size=len(df))
    set_noise = r < 0.00045
    for i in np.argwhere(set_noise):
        c = round(uniform.rvs(30))
        c = min(c, len(set_noise)-i[0])
        set_noise[i[0]:i[0]+c] = True
    
    label = pd.Series(set_noise, index = df.index).rename("label")
    
    dfErr = df.copy()
    dfErr[label] = noise[label]
    assert (df[~label]==dfErr[~label]).all().all()
    assert not (df[label]==dfErr[label]).any().any()
    return label, dfErr

combine_file_path = Path("../dataset/M01-12.csv") # 合併的資料集

err_dir = Path("../dataset/o/")
data_fn = err_dir / "data.csv"
label_fn = err_dir / "eLabel.csv"
day_label_fn = err_dir / "day_Label.csv"
    
if __name__ == "__main__":
    # 讀取step1合併的資料集，並另第一欄(timestamp)作為index
    df_merge = pd.read_csv(combine_file_path, index_col = 0)
    
    label, dfErr = dataModify_v1(df_merge)
    
    # 若資料夾不存在，創建它
    if not err_dir.is_dir():
        err_dir.mkdir()
    
    # save 以及read 修改的資料
    dfErr.to_csv(data_fn, encoding='utf_8_sig')
    errRead = pd.read_csv(data_fn, index_col = 0)
    
    # save 以及read 是否修改的時間戳對應表
    label.to_csv(label_fn)
    labelRead = pd.read_csv(label_fn, index_col = 0)
    # print(labelRead)
    
    # 取得時間戳的date與i
    timeRead = pd.to_datetime(labelRead.index.to_series())
    date = timeRead.apply(lambda _ : _.date()).rename("date")
    i = timeRead.apply(lambda _ : 60 * _.hour + _.minute).rename("i")
    
    # 計算每15分鐘以及每天的是否修改標籤
    label_15m = pd.concat([timeRead, date, i // 15, label], axis=1)
    G_15m = label_15m.groupby(["date", "i"]).label.any()
    print(sum(G_15m)/len(G_15m), "(", sum(G_15m), "/", len(G_15m), ")")
    
    label_day = pd.concat([timeRead, date, label], axis=1)
    G_day = label_day.groupby("date").label.any()
    # print(sum(G_day)/len(G_day), "(", sum(G_day), "/", len(G_day), ")")
    
    G_day.to_csv(day_label_fn)
    t_G_day = pd.read_csv(day_label_fn, index_col = 0)["label"]
    print(sum(t_G_day)/len(t_G_day), "(", sum(t_G_day), "/", len(t_G_day), ")")
    # print(t_G_day)
    