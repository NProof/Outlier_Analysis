# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

combine_file_path = Path("../dataset/M01-12.csv")
output_dir = Path("../dataset/combine_version")

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
    
if __name__ == "__main__":
    # 讀取step1合併的資料集，並另第一欄(timestamp)作為index
    df_merge = pd.read_csv(combine_file_path, index_col = 0)
    
    # 因timestamp讀取後不會自動轉型成datetime，而是str型別，故轉型
    temps_step2 = pd.to_datetime(df_merge.index.to_series())
    
    # 轉換temps成兩個欄位(date, i)，date是日期，i是每天第i個分鐘的意思
    dates = temps_step2.apply(lambda _ : _.date()).rename("date")
    i = temps_step2.apply(lambda _ : 60 * _.hour + _.minute).rename("i")
    
    # 若資料夾不存在，創建它
    if not output_dir.is_dir():
        output_dir.mkdir()
    
    # 將單一特徵轉換型狀，以date作為index、i為欄位的表
    # (只產生 COVERT_COL 中的表)
    for col, out_file_name in COVERT_COL.items():
        print('{:^25}->{:^20}'.format(col, out_file_name))
        cur_ser = df_merge[col].rename("val")
        _ = pd.concat([dates, i, cur_ser], axis=1)
        df_ser = _.pivot_table(index = "date", columns = "i", values = "val")
        df_ser.to_csv(output_dir / (out_file_name + ".csv"))
        