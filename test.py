# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import uniform

from LAKE import tf, keras, LVAE

from sklearn.neighbors import KernelDensity
from scipy.spatial import distance
from scipy.stats import zscore

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

combine_file_path = Path("../dataset/M01-12.csv") # 合併的資料集

# 人為修改資料集的資料夾與檔案路徑
err_dir = Path("../dataset/err_v2-0006/")
data_fn = err_dir / "data.csv"
label_fn = err_dir / "eLabel.csv"
day_label_fn = err_dir / "day_Label.csv"

delta = 15
iteDay = 1440/delta

def DP(f_df, timestamp):
    date = timestamp.apply(lambda _ : _.date())
    hour = timestamp.apply(lambda _ : _.hour)
    minute = timestamp.apply(lambda _ : _.minute)
    
    avg_FDate = f_df.groupby(
        [
         lambda i : date[i], 
         lambda i : ( hour[i] * 60 + minute[i] ) // delta
        ]
    ).mean()
    print(avg_FDate)
    
    countDay1 = avg_FDate.groupby(level=0).apply(lambda _ : len(_))
    avg_FDate_drop = avg_FDate.drop((
        countDay1.loc[countDay1 != iteDay]
        ).index)
    countDay2 = avg_FDate_drop.groupby(level=0).apply(lambda _ : len(_))
    assert all(countDay2 == iteDay) # sum(countDay2 != iteDay) == 0
    unitFrames = avg_FDate_drop.unstack(level=1).dropna() # unitFrames.stack(level=1) # [Redo -> avg_FDate_drop]
    return unitFrames

from datetime import datetime
from datetime import timedelta

def showTimeSeries(k_test):
    fig = plt.subplots(1, figsize=(70, 15))
    plt.plot(k_test.index.to_series(), k_test.densityln)
    plt.show()
    
if __name__ == "__main__":
    
    # 讀取修改資料拆分的檔案
    data_err = pd.read_csv(data_fn, index_col = 0)
    # print(data_err)
    label_err = pd.read_csv(label_fn, index_col = 0)
    # print(label_err)
    labelD_err = pd.read_csv(day_label_fn, index_col = 0)["label"]
    # print(labelD_err)
    print('{:.6f} ({}/{})'.format(
        sum(labelD_err)/len(labelD_err),
        sum(labelD_err),
        len(labelD_err)
    ))
    
    tf.keras.backend.set_floatx('float64')
    tf.keras.backend.floatx()
    tf.debugging.enable_check_numerics()
    
    df_raw = pd.read_csv('../dataset/M01-12.csv', index_col = 0)
    df_raw.index = pd.to_datetime(df_raw.index.to_series())
    timestamp = df_raw.index.to_series()
    # print(df_raw)
    df_raw_back = df_raw.copy()

    f_df = df_raw['冰機Q(sum)']
    
    train = DP(f_df, timestamp)
    
    model = LVAE(latent_dim = 6, fDim = train.shape[1])
    # model.built = True
    # model.load_weights('../dataset/err_v2-0006/err_v3-0006.h5')
    # model.built = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5))
    history = model.fit(train, epochs=400, batch_size=8)
    model.save_weights("../dataset/err_v2-0006/err_v3-0006_2.h5")

    c = model.transferESpace(train)
    
    data_err.index = pd.to_datetime(data_err.index.to_series())
    label_err.index = pd.to_datetime(label_err.index.to_series())
    labelD_err.index = pd.to_datetime(labelD_err.index.to_series())
    
    test = DP(data_err['冰機Q(sum)'], label_err.index.to_series())
    c_test = model.transferESpace(test)
        
    for h in [5.0]: #[5**i for i in np.linspace(start = -5, stop = 2, num = 50)]: # in [5.0]:
        # h = 5.0
        kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(c)
        k = pd.DataFrame(kde.score_samples(c), index=train.index, columns=['densityln'])
        # showTimeSeries(k)
        
        k_test = pd.DataFrame(kde.score_samples(c_test), index=test.index, columns=['densityln'])
        # showTimeSeries(k_test)
        
        k_test['z'] = (pd.DataFrame(k_test.densityln) - k_test.densityln.mean()) / k_test.densityln.std()
        
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 16))
        ax1 = sns.violinplot(ax=ax, orient='v', y=k.densityln)
        fig.text(0.5, 0.04, "h = " + str(h), ha='center', fontsize = 20)
        ax1.set_ylabel("density probability", fontsize = 20)
        ax1.set_xlabel("traing data after KDE", fontsize = 20)
        plt.show()
        
        # fig, axes = plt.subplots(1, 2, sharex=True, figsize=(12, 8))
        # ax1 = sns.violinplot(ax=axes[0], orient='v', y=k.densityln)
        # ax2 = sns.violinplot(ax=axes[1], orient='v', y=k_test.z)
        # fig.text(0.5, 0.04, "h = " + str(h), ha='center', fontsize = 20)
        # ax1.set_ylabel("density probability", fontsize = 20)
        # ax2.set_ylabel("density probability with z-scores", fontsize = 20)
        # plt.show()
        
        metric_Day = pd.DataFrame(data={
            'ground_true' : ~labelD_err.loc[test.index], 
            'predict' : k_test.z
            })
        
        fpr, tpr, thresholds = roc_curve(metric_Day.ground_true, metric_Day.predict, pos_label=1)
        print("h = " + str(h), ", AUC_ROC", roc_auc_score(metric_Day.ground_true, metric_Day.predict))
        print(confusion_matrix(metric_Day.ground_true, metric_Day.predict > -3))
        print(precision_recall_fscore_support(metric_Day.ground_true, metric_Day.predict > -3))
    
    ####   ####   ####    ####
    # print( metric_Day[metric_Day.predict<-3] )
    
    ########
    # fpr, tpr, thresholds = roc_curve(metric_Day.ground_true, metric_Day.predict, pos_label=1)
    # print( roc_auc_score(metric_Day.ground_true, metric_Day.predict) )
    
    # plt.plot(fpr, tpr)
    # plt.show()
    
    # print(confusion_matrix(metric_Day.ground_true, metric_Day.predict > -3))
    # print(precision_recall_fscore_support(metric_Day.ground_true, metric_Day.predict > -3))
    
        