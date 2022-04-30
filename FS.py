from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

combine_file_path = Path("../dataset/M01-12.csv")

df_raw = pd.read_csv(combine_file_path, index_col = 0)
df_raw.index = pd.to_datetime(df_raw.index.to_series())
timestamp = df_raw.index.to_series()

# [
#  df_raw.columns.get_loc(col_name) for col_name 
#  in ['冰機Vb', '冷卻水泵Q(sum)', 'temp1(冷卻水出水溫度#1)(Tcwst1)']
# ] # [1, 30, 45]

rData = df_raw.iloc[:, [1, 30, 45]]

x = rData.iloc[:, 0].to_numpy()
y = rData.iloc[:, 1].to_numpy()
# np.expand_dims(x.to_numpy(), axis=1)

m1 = 1 - abs(rData.corr())
m1.iat[0, 1] # 0.5212417064935069

print( np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y ) )

lr = LinearRegression()
lr.fit(np.expand_dims(x, axis=1), y)
w = lr.coef_
print(w)

error = np.linalg.norm(np.expand_dims(x, axis=1).dot(w) - y, ord=2) ** 2

orp1 = np.concatenate((rData.iloc[:, [0, 2]], np.ones([len(x), 1])), axis=1)

v1 = np.ones( [len(x), 1] )
test = np.append(x.iloc[:, [0]].to_numpy(), v1, axis=1)
print( np.linalg.lstsq(test, x.iloc[:, 1].to_numpy()) )

lr2 = LinearRegression()
lr2.fit(test, y)
print( lr2.coef_ )
