# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 14:29:13 2021

@author: User
"""

import pandas as pd
import datetime
from statistics import mean

import data_processing
from sklearn import preprocessing
from LAKE import tf, keras, LVAE

delta = 15
iteDay = 1440//delta
# h = 5

def dataPreProcessing(f_df, time = None):
    if time is None:
        time = data_processing.convertTime(f_df.index.to_series())
    
    date = time['d'].loc[f_df.index]
    hour = time['h'].loc[f_df.index]
    minute = time['m'].loc[f_df.index]
    
    avg_FDate = f_df.groupby(
        [
          lambda i : date[i], 
          lambda i : (hour[i] * 60 + minute[i]) // delta
        ]).mean()
    
    unitFrames = pd.DataFrame(
        {
            avg_FDate.index[i]: avg_FDate.iloc[i:i+iteDay].reset_index(drop=True).reset_index(drop=True).unstack(level=0)
            for i in range(len(avg_FDate)-iteDay) # range(10)
        }
    )
    return unitFrames

def meanErr(test, reconstruction_test):
    errtest = test - reconstruction_test.numpy()
    
    dfmap = dict()
    
    for date, row in errtest[:].T.iteritems():
        for i, r in row.iteritems():
            target = (
                i[0], 
                datetime.datetime(*date[0].timetuple()[:6]) + datetime.timedelta(hours=0.5) * (i[1] + date[1])
            )
            if target not in dfmap:
                dfmap[target] = list()
            dfmap[target].append(r)
            
    dfmean = dict()        
    
    for k, v in dfmap.items():
        dfmean[k] = mean(v)
        
    a = pd.DataFrame(dfmean.values(), dfmean.keys())
    return a
    

if __name__ == '__main__':
    
    tf.keras.backend.set_floatx('float64')
    tf.keras.backend.floatx()
    tf.debugging.enable_check_numerics()
    
    history = pd.read_csv('./dataset/Merge.csv').dropna()
    # cut parition
    # history = history.iloc[:25000]
    history = data_processing.deduplicate(history, subject='時間')
    # add timestamp
    history['時間'] = history['時間'].apply(lambda _ : pd.to_datetime(_))
    time_h = data_processing.convertTime(history['時間'])
    history = history.set_index("時間")
    
    history = history[["冰機Q(sum)"]]
    
    # transformer = preprocessing.RobustScaler().fit(history)
    transformer = preprocessing.StandardScaler().fit(history)
    
    f_df1 = pd.DataFrame(transformer.transform(history), index=history.index, columns=history.columns)
    train = dataPreProcessing(f_df1, time_h).T
    # f_df2 = pd.DataFrame(transformer.inverse_transform(f_df1), index=f_df1.index, columns=f_df1.columns)
    
    # PART TEST WITH MODIFIED ERROR

    err = pd.read_csv('./dataset/err-0001.csv').dropna()
    # cut parition
    # err = err.iloc[:25000]
    err = data_processing.deduplicate(err, subject='time')
    # add timestamp
    err['time'] = err['time'].apply(lambda _ : pd.to_datetime(_))
    time_e = data_processing.convertTime(err['time'])
    err = err.set_index('time')
    
    label = err[["label"]]
    err = err[["冰機Q(sum)"]]
    
    e_df1 = pd.DataFrame(transformer.transform(err), index=err.index, columns=err.columns)
    test = dataPreProcessing(e_df1, time_h).T
    # e_df2 = pd.DataFrame(transformer.inverse_transform(e_df1), index=e_df1.index, columns=e_df1.columns)
    
    file_name_save = "weightsModel/Q0106-02"
    
    nums_models = 7
    models = []
    list_merr = []
    
    for i in range(nums_models):
        model = LVAE(latent_dim = 6, fDim = train.shape[1])
        # model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5))
        # history = model.fit(train, epochs=25, batch_size=8)
        # model.save_weights(file_name_save+"_"+str(i))
        model.load_weights(file_name_save+"_"+str(i))
        models.append(model)
        
        _, _, _, _, _, z_test = models[i].encoder.predict(test)
        _, _, _, reconstruction_test = models[i].decoder(z_test)
    
        ret = meanErr(test, reconstruction_test)
        list_merr.append(ret)
        
    del reconstruction_test
    ### ### ### ### ### ### ### ### ### ### ### ### ###
        
    c = pd.concat(list_merr, keys=range(nums_models), axis=1, ignore_index=True)
    arg_opt = pd.DataFrame([abs(r).argmin() for i, r in c.iterrows()], index=c.index)
    # c['arg'] = arg_opt
    opt = pd.Series([r.loc[arg_opt.loc[i]].values[0] for i, r in c.iterrows()], index=c.index)
    del arg_opt
    
    opt_model = (c**2).mean().argmin()
    
    