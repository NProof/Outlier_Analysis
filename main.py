# -*- coding: utf-8 -*-

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from sklearn.neighbors import KernelDensity
# from scipy.stats import zscore
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support

from LAKE import LVAE
from feature_selection import PFA

delta = 15
iteDay = 1440/delta
h = 5

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

def convertTime(timestamp):
    date = timestamp.apply(lambda _ : _.date())
    hour = timestamp.apply(lambda _ : _.hour)
    minute = timestamp.apply(lambda _ : _.minute)
    time = pd.DataFrame(data={'d':date.values, 'h':hour.values, 'm':minute.values}, index=timestamp)
    return time
    
def dataPreProcessing(f_df, time):
    if time is None:
        time = convertTime(f_df_raw.index)
    
    date = time['d'].loc[f_df.index]
    hour = time['h'].loc[f_df.index]
    minute = time['m'].loc[f_df.index]
    
    avg_FDate = f_df.groupby(
        [
          lambda i : date[i], 
          lambda i : ( hour[i] * 60 + minute[i] // delta )
        ]).mean()
    
    countDay1 = avg_FDate.groupby(level=0).apply(lambda _ : len(_))
    avg_FDate_drop = avg_FDate.drop((
        countDay1.loc[countDay1 != iteDay]
        ).index)
    countDay2 = avg_FDate_drop.groupby(level=0).apply(lambda _ : len(_))
    assert all(countDay2 == iteDay) # sum(countDay2 != iteDay) == 0
    unitFrames = avg_FDate_drop.unstack(level=1).dropna() # unitFrames.stack(level=1) # [Redo -> avg_FDate_drop]
    
    return unitFrames

def trainModel(unitFrames, file_name_save):
    model = LVAE(latent_dim = 6, fDim = unitFrames.shape[1])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5))
    history = model.fit(unitFrames, epochs=324, batch_size=8)
    
    model.save_weights(file_name_save)
    model = LVAE(latent_dim = 6, fDim = unitFrames.shape[1])
    model.load_weights(file_name_save)
    return model, history

if __name__ == '__main__':
    
    tf.debugging.enable_check_numerics()
    tf.keras.backend.set_floatx('float64')
    tf.keras.backend.floatx()
    
    df_raw = pd.read_csv('./dataset/Merge.csv')
    
    ser_dup = ~df_raw['時間'].duplicated()
    f_df_raw = df_raw[ser_dup].reset_index(drop=True).dropna()
    timestamp = f_df_raw.pop('時間').apply(lambda _ : pd.to_datetime(_))
    f_df_raw.index = timestamp
    # print(f_df_raw)
    
    column_names = featureSelecting(f_df_raw)
    
    time = convertTime(timestamp)
    # print(time)
    
    f_df = pd.DataFrame(f_df_raw[column_names], index=f_df_raw.index)
    
    transformer1 = preprocessing.RobustScaler().fit(f_df)
    f_df[:] = transformer1.transform(f_df)
    
    train = dataPreProcessing(f_df, time)
    
    f_df[:] = transformer1.inverse_transform(f_df)
    print(f_df)
    model, history = trainModel(train, './weightsModel/1209-1')
    
    c = model.transferESpace(train)
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(c)
    k = pd.DataFrame(kde.score_samples(c), index=train.index, columns=['densityln'])
    
    # ---------------------------------------------
    
    err_raw = pd.read_csv('./dataset/err-0001.csv')
    
    ser_dup_err = ~err_raw['time'].duplicated()
    f_err_raw = err_raw[ser_dup_err].reset_index(drop=True).dropna()
    timestamp_err = f_err_raw.pop('time').apply(lambda _ : pd.to_datetime(_))
    f_err_raw.index = timestamp_err
    
    fig, axs = plt.subplots(2, figsize=(100, 15))
    axs[0].plot(timestamp, f_df_raw['冰機Q(sum)'], 'b.')
    axs[1].plot(f_err_raw.index.to_series()[~f_err_raw.label], f_err_raw[~f_err_raw.label]['冰機Q(sum)'], 'g.')
    axs[1].plot(f_err_raw.index.to_series()[f_err_raw.label], f_err_raw[f_err_raw.label]['冰機Q(sum)'], 'r.')
    plt.show()
    
    time_err = convertTime(timestamp_err)
    # print(time_err)
    
    tlErr = pd.DataFrame({'time' : f_err_raw.index.to_series(), 'label' : f_err_raw['label'].values})
    
    f_err = f_err_raw[column_names]
    
    f_err[:] = transformer1.transform(f_err)
    test = dataPreProcessing(f_err, time_err)
    f_err[:] = transformer1.inverse_transform(f_err)
    
    unstackLabel = tlErr.groupby(lambda i: time_err['d'].loc[i]).apply(lambda i : i['label'].any())
    
    c_test = model.transferESpace(test)
    k_test = pd.DataFrame(kde.score_samples(c_test), index=test.index, columns=['densityln'])
    
    k_test['z'] = (pd.DataFrame(k_test.densityln) - k.densityln.mean()) / k_test.densityln.std()
    
    metric_Day = pd.DataFrame(data={
        'ground_true' : ~unstackLabel.loc[test.index], 
        'predict' : k_test.z
        })
    
    precision_recall_fscore_support(metric_Day.ground_true, metric_Day.predict > -3)
    
    fpr, tpr, thresholds = roc_curve(metric_Day.ground_true, metric_Day.predict, pos_label=1)
    print( roc_auc_score(metric_Day.ground_true, metric_Day.predict) )
    
    plt.plot(fpr, tpr)
    plt.show()