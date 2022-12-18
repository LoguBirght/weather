import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import math
#%matplotlib inline
warnings.filterwarnings('ignore')


#station 3
#time
#pm2.5

#基隆觀測站=train_data
#淡水觀測站=test_data

dtype = {
    "station": str,
    "AMB_TEMP": int,
    "CH4": int,
    "CO": float,
    "NMHC": float,
    "NO": float,
    "NO2": int,
    "NOx": int,
    "O3": int,
    "PH_RAIN": str,
    "PM10": int,
    "PM2.5": int,
    "RAINFALL": str,
    "RAIN_COND": str,
    "RH": str,
    "SO2": int,
    "THC": float,
    "UVB": int,
    "WD_HR": int,
    "WIND_DIREC": int,
    "WIND_SPEED": float,
    "WS_HR": float
}
#| # indicates invalid value by equipment inspection
#| * indicates invalid value by program inspection
#| x indicates invalid value by human inspection
#| NR indicates no rainfall
#| blank indicates no data

def clean_data(elem):
    elem = str(elem)
    if elem is None or elem is '':
        return np.NaN
    if any(x in elem for x in ["#", "*", "x"]):
        return np.NaN
    if any(x in elem for x in [".", "e"]):
        return float(elem)
    if elem in 'NR':
        return -1
    return int(elem)

converters = {
    "AMB_TEMP": clean_data,
    "CH4": clean_data,
    "CO": clean_data,
    "NMHC": clean_data,
    "NO": clean_data,
    "NO2": clean_data,
    "NOx": clean_data,
    "O3": clean_data,
    "PH_RAIN": clean_data,
    "PM10": clean_data,
    "PM2.5": clean_data,
    "RAINFALL": clean_data,
    "RAIN_COND": clean_data,
    "RH": clean_data,
    "SO2": clean_data,
    "THC": clean_data,
    "UVB": clean_data,
    "WD_HR": clean_data,
    "WIND_DIREC": clean_data,
    "WIND_SPEED": clean_data,
    "WS_HR": clean_data
}

df = pd.read_csv('./Air_quality.csv', dtype = dtype, converters = converters, parse_dates = ['time'])
df.dtypes
df = df.drop(columns=['AMB_TEMP','CH4','CO','NMHC','NMHC','NO','NO2','NOx','O3','PH_RAIN','PM10','RAINFALL','RAIN_COND','RH','SO2','THC','UVB','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR'])
#-----------------------------補齊null----------------------------------
mean_pm=df[['station','PM2.5']].groupby('station').mean()
mapping=dict(zip(mean_pm.index,mean_pm['PM2.5']))
for i in range(len(df['PM2.5'].isnull())):
    if df['PM2.5'].isnull()[i]:
        df.loc[i,'PM2.5']=mapping.get(df['station'][i])
#-------
#df.plot(figsize=(18,5),x='time',y='PM2.5')
#plt.show()
#-----------------------------取出我們要的站名&&補齊null----------------------------------
#test:淡水
#train:基隆 Keelung,觀音 Guanyin,
#
levels = df['station'].unique()

train_data=df[df['station'] != 'Tamsui']

test_data=df[df['station'] == 'Tamsui']
#-----------test --------
test_data.plot.line(x='time',y='PM2.5')
plt.title("Tamsui Pm2.5 in years")
plt.ylabel("PM2.5")
plt.xlabel("time")
#plt.show()
#--------------預測模型------------------
#df2=df
test_data.head(10)
pre = test_data.drop(columns=['PM2.5'])
pre['PM2.5']=np.nan
pre.head(10)
#----------------------

mean_ppm=train_data[['time','PM2.5']].groupby('time').mean()
mapping_p=dict(zip(mean_ppm.index,mean_ppm['PM2.5']))
for i in range(len(pre)):
    pre.loc[113712+i,'PM2.5'] = mapping_p.get(train_data['time'][i])
pre.head(10)

pre.plot.line(x='time',y='PM2.5')
plt.title("predictions of Tamsui Pm2.5 in years")
plt.ylabel("PM2.5")
plt.xlabel("time")
plt.show()
