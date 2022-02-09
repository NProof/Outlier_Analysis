# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import uniform

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
    
# 為了方便管理，每個月的資料集中於特定目錄 (origin_dir_path)
origin_dir_path = Path("../dataset/origin_version")
combine_file_path = Path("../dataset/M01-12.csv") # 合併的資料集
output_dir = Path("../dataset/combine_version") # 合併的單一特徵資料集

# 人為修改資料集的資料夾與檔案路徑
err_dir = Path("../dataset/o/")
data_fn = err_dir / "data.csv"
label_fn = err_dir / "eLabel.csv"
day_label_fn = err_dir / "day_Label.csv"

# 資料中 [時間] 所有的格式
formats = ["%Y/%m/%d %p %I:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/ %m / %d %p %I:%M:%S"]

COVERT_COL = {
    "冰機Vac" : "chiller_load",
    "temp1(冷卻水回水溫度#1)(Tcwrt1)" : "chr_temp",
    "temp1(冷卻水出水溫度#1)(Tcwst1)" : "chs_temp",
    "冰水泵Vac" : "chw_pump",
    "temp1(冰水回水溫度#1)(Tchwrt1)" : "chwr_temp",
    "temp1(冰水出水溫度#1)(Tchwst1)" : "chws_temp",
    "冷卻水塔Vac" : "cooling_tower",
    "冷卻水泵Vac" : "cw_pump",
    "外氣濕度" : "outside_air_hum",
    "外氣溫度" : "outside_air_temp"
    }
    
# step1
def combineData():
    # 確認目錄是否合法
    assert(origin_dir_path.exists())
    
    df_step1 = CombineXLSX(origin_dir_path)
    time_str = df_step1.pop("時間")
    temps_step1 = str2Datetime(time_str)
    
    # 在儲存 df_step1 (.csv file) 之前
    # 把 index <- temps_step1 / columns name 去除 "\n"
    df_step1.index = temps_step1
    df_step1.columns = df_step1.columns.to_series().apply(lambda _ : _.replace("\n", ""))
    df_step1.to_csv(combine_file_path, encoding='utf_8_sig')
    
# step2
def splitIndData():
    # 讀取step1合併的資料集，並另第一欄(timestamp)作為index
    df_merge = pd.read_csv(combine_file_path, index_col = 0)
    
    # 因timestamp讀取後不會自動轉型成datetime，而是str型別，故轉型
    temps_step2 = pd.to_datetime(df_merge.index.to_series())

    # 轉換temps成兩個欄位(date, i)，date是日期，i是每天第i個分鐘的意思
    date = temps_step2.apply(lambda _ : _.date()).rename("date")
    i = temps_step2.apply(lambda _ : 60 * _.hour + _.minute).rename("i")
    
    # 若資料夾不存在，創建它
    if not output_dir.is_dir():
        output_dir.mkdir()
    
    # 將單一特徵轉換型狀，以date作為index、i為欄位的表
    # (只產生 COVERT_COL 中的表)
    for col, out_file_name in COVERT_COL.items():
        print('{:^25}->{:^20}'.format(col, out_file_name))
        cur_ser = df_merge[col].rename("val")
        _ = pd.concat([date, i, cur_ser], axis=1)
        df_ser = _.pivot_table(index = "date", columns = "i", values = "val")
        df_ser.to_csv(output_dir / (out_file_name + ".csv"))
    
# step3
def modifyData():
    # [同] 讀取step1合併的資料集，並另第一欄(timestamp)作為index
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
    
# (Q1) 資料中的時間格式不一致
    # (A1) str2Datetime(str_series)

# step1 : Combine all of data into one file.
# step2 : Split into individual feature data
# step3 : 產生人為修改的資料集

# later : 
    # 1) metric design
    # 2) predict algorithm
    
if __name__ == "__main__":
    combineData()
    splitIndData()
    modifyData()
    