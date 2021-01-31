#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:15:13 2021

@author: zyhhhh
"""

import pandas as pd
import datetime
import matplotlib.pyplot as plt
path='/Users/zhaoyuheng/Downloads/6980/20170303_01F0339S_to_01F0413S.csv'
df = pd.read_csv(path)
df.iloc[:,8] = pd.to_datetime(df.iloc[:,8])
df.iloc[:,9] = pd.to_datetime(df.iloc[:,9])
minutes = list(range(60))
hours = list(range(24))
data=[]

for each_hour in hours:
    for each_minute in minutes:
        df2 = df.loc[(df.iloc[:,8].dt.minute == each_minute) & (df.iloc[:,8].dt.hour == each_hour)]
        df3 = df.loc[(df.iloc[:,9].dt.minute == each_minute) & (df.iloc[:,9].dt.hour == each_hour)]
        if len(df2.iloc[:,1]):
            data.append([each_hour, each_minute, len(df2.iloc[:,1]),len(df3.iloc[:,1])])
            
data=pd.DataFrame(data)
start=data.loc[0:360:,2].sum()-data.loc[0:360:,3].sum()
# time start 6:00 am, calculate acc from 0:00.

flux=[]
acc1=[]
for i in range(361,1381,1):
    flux.append(data.loc[i,3])
    temp=start+data.loc[i,2]-data.loc[i,3]
    acc1.append(temp)
    start+=(data.loc[i,2]-data.loc[i,3])

plt.scatter(acc1,flux,s=5)