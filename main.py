# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing

import matplotlib.pyplot as plt

import LAKE
import tensorflow as tf
from tensorflow import keras
from feature_selection import PFA

# # if __name__ == '__main__':

tf.keras.backend.set_floatx('float64')
tf.keras.backend.floatx()

df_raw = pd.read_csv('./dataset/Merge.csv')
non_dup_df_raw = df_raw[~df_raw['時間'].duplicated()].reset_index(drop=True)
non_dup_f_df_raw = non_dup_df_raw.dropna()

f_df_raw = non_dup_f_df_raw.copy()
qsum = f_df_raw['冰機Q(sum)']

timestamp = f_df_raw.pop('時間').apply(lambda _ : pd.to_datetime(_))
date = timestamp.apply(lambda _ : _.date())
hour = timestamp.apply(lambda _ : _.hour)
minute = timestamp.apply(lambda _ : _.minute)
time = pd.DataFrame(data={'d':date.values, 'h':hour.values, 'm':minute.values}, index=timestamp)
# print(time)
f_df_raw.index = timestamp
# print(f_df_raw)

X = f_df_raw
pfa = PFA(n_features=10)
pfa.fit(X)

# To get the transformed matrix
X = pfa.features_

# To get the column indices of the kept features
column_indices = pfa.indices_

delta = 15
iteDay = 1440/delta

f_df = pd.DataFrame(f_df_raw[f_df_raw.columns[column_indices]], index=f_df_raw.index)

transformer1 = preprocessing.RobustScaler().fit(f_df)
f_df[:] = transformer1.transform(f_df)

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

f_df[:] = transformer1.inverse_transform(f_df)
print(f_df)

model = LAKE.VAE(latent_dim = 6, fDim = unitFrames.shape[1])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5))
history = model.fit(unitFrames, epochs=324, batch_size=8)

model.save_weights('./weightsModel/1209-1')
model = LAKE.VAE(latent_dim = 6, fDim = unitFrames.shape[1])
model.load_weights('./weightsModel/1209-1')

from sklearn.neighbors import KernelDensity
from scipy.spatial import distance
# from scipy.stats import zscore

train = unitFrames

_, _, _, _, _, z = model.encoder.predict(train)
_, _, _, reconstruction = model.decoder(z)
r = np.array(
    [
     [distance.euclidean(i, j), distance.cosine(i, j)]
     for i, j in zip(tf.convert_to_tensor(train), reconstruction)])

c = np.concatenate([z, r], axis=1)

h = 5
kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(c)

k = pd.DataFrame(kde.score_samples(c), index=train.index, columns=['densityln'])

err_raw0 = pd.read_csv('./dataset/err-0001.csv')
err_raw = err_raw0.copy()

timeErr_raw = err_raw['time'].apply(lambda _ : pd.to_datetime(_))

non_duplicated = ~timeErr_raw.duplicated()
err = err_raw[non_duplicated].dropna().reset_index(drop=True)
timeErr = timeErr_raw[non_duplicated].dropna().reset_index(drop=True)

date_err = timeErr.apply(lambda _ : _.date())
hour_err = timeErr.apply(lambda _ : _.hour)
minute_err = timeErr.apply(lambda _ : _.minute)

time = pd.DataFrame(data={'d':date_err.values, 'h':hour_err.values, 'm':minute_err.values}, index=timeErr)
# print(time)
tlErr = pd.DataFrame({'time' : timeErr.values, 'label' : err['label'].values})

f_err = err[f_df_raw.columns[column_indices]]

delta = 15
iteDay = 1440/delta

groupedErr = f_err.groupby(
    [
     lambda i: date_err[i], 
     lambda i : (60 * hour_err[i] + minute_err[i]) // 15
     ]
    )
meanErr = groupedErr.mean()

countDay1 = meanErr.groupby(level=0).apply(lambda _ : len(_))
meanErr_drop = meanErr.drop((
    countDay1.loc[countDay1 != iteDay]
    ).index)

countDay2 = meanErr_drop.groupby(level=0).apply(lambda _ : len(_))
assert all(countDay2 == iteDay) # sum(countDay2 != iteDay) == 0
unstackMErr = meanErr_drop.unstack(level=1).dropna() # unstackMErr.stack(level=1) # [Redo -> avg_FDate_drop]
print(unstackMErr)

unstackLabel = tlErr.groupby(lambda i: time['d'].iloc[i]).apply(lambda i : i['label'].any())

test = unstackMErr

_, _, _, _, _, z_test = model.encoder.predict(test)
_, _, _, reconstruction_test = model.decoder(z_test)
r_test = np.array(
    [
     [distance.euclidean(i, j), distance.cosine(i, j)]
     for i, j in zip(tf.convert_to_tensor(test), reconstruction_test)])

c_test = np.concatenate([z_test, r_test], axis=1)

k = pd.DataFrame(kde.score_samples(c), index=train.index, columns=['densityln'])
k_test = pd.DataFrame(kde.score_samples(c_test), index=test.index, columns=['densityln'])
k_test['z'] = (pd.DataFrame(k_test.densityln) - k.densityln.mean()) / k_test.densityln.std()

metric_Day = pd.DataFrame(data={
    'ground_true' : ~unstackLabel.loc[test.index], 
    'predict' : k_test.z
    })

fig, axs = plt.subplots(2, figsize=(100, 15))
axs[0].plot(timestamp, f_df_raw['冰機Q(sum)'], 'b.')
axs[1].plot(timeErr[~err.label], f_err[~err.label]['冰機Q(sum)'], 'g.')
axs[1].plot(timeErr[err.label], f_err[err.label]['冰機Q(sum)'], 'r.')
plt.show()

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(metric_Day.ground_true, metric_Day.predict > -3)

