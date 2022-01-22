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
    df_merge = pd.read_csv(combine_file_path)
    df_merge = df_merge.set_index("timestamp")
    
    temps_step2 = pd.to_datetime(df_merge.index.to_series())
    dates = temps_step2.apply(lambda _ : _.date())
    hours = temps_step2.apply(lambda _ : _.hour)
    minutes = temps_step2.apply(lambda _ : _.minute)
    i = 60 * hours + minutes
    
    if not output_dir.is_dir():
        output_dir.mkdir()
    
    for col, out_file_name in COVERT_COL.items():
        print(col, "->", out_file_name, end="")
        cur_ser = df_merge[col]
        _ = pd.concat([dates, i, cur_ser], keys=["date", "i", "val"], axis=1)
        df_ser = _.pivot_table(index = "date", columns = "i", values = "val")
        df_ser.to_csv(output_dir / (out_file_name + ".csv"))
