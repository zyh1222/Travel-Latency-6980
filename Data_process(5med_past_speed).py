#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:30:26 2021

@author: zyhhhh
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from itertools import chain

def select_raw_latency_by_segment(df_input, entrance_sensor, exit_sensor):
    df = df_input.xs((entrance_sensor, exit_sensor), level=('Entrance', 'Exit'))
    return pd.DataFrame(df['Latency_Raw'])

# These pickle files are developed by multi_process_extract.py
df1 = pd.read_pickle('/Users/zhaoyuheng/Downloads/6980/L201701.pickle')
df2 = pd.read_pickle('/Users/zhaoyuheng/Downloads/6980/L201704.pickle')
df3 = pd.read_pickle('/Users/zhaoyuheng/Downloads/6980/L201707.pickle')
df4 = pd.read_pickle('/Users/zhaoyuheng/Downloads/6980/L201710.pickle')
df_input1 = df1
df_input2 = df2
# df_input2.dropna(axis=0, how='any', inplace=True)
df_input3 = df3
df_input4 = df4
entrance_sensor, exit_sensor = '01F0339S', '01F0509S'
df1 = select_raw_latency_by_segment(df_input1, entrance_sensor, exit_sensor)
df2 = select_raw_latency_by_segment(df_input2, entrance_sensor, exit_sensor)
df3 = select_raw_latency_by_segment(df_input3, entrance_sensor, exit_sensor)
df4 = select_raw_latency_by_segment(df_input4, entrance_sensor, exit_sensor)
df1 = df1[df1.index.date!=(df1.index.date[-1])]
df2 = df2[df2.index.date!=(df2.index.date[-1])]
df3 = df3[df3.index.date!=(df3.index.date[-1])]
df4 = df4[df4.index.date!=(df4.index.date[-1])]
df = pd.concat([df1,df2,df3,df4])

# get rolling median latency with arrival time
def get_rolling_median_latency(df_latency_raw,rolling_window=5):
    df = pd.DataFrame(df_latency_raw.copy())
    rolling_column = df_latency_raw.name
    Que = []
    min_med = [np.nan]*(rolling_window-1)
    for ni,li in enumerate(df[rolling_column]):
        Que.append(li)
        if len(Que)<rolling_window:
            continue
        #total median
        mylist=np.array(list(chain(*Que)))
        mylist=mylist[mylist>=0]
        min_med.append(np.median(mylist))
        Que = Que[1:]
    df[rolling_column]=min_med
    return df[rolling_column]
# data=get_rolling_median_latency(df['Latency_Raw'])
# data = data[(data.index.hour<23) & (data.index.hour>=6)]
# data = data[(data.index.year==2017)]
# data.to_csv('/Users/zhaoyuheng/Downloads/OneDrive_1_6-15-2020/Data/5_med_latency1.csv')

def get_rolling_latency(df_latency_raw, rolling_windows=5):
    df = pd.DataFrame(df_latency_raw.copy())
    rolling_column = df_latency_raw.name
    # sum all the elements in the list of raw latency
    df['sum'] = [sum(i) for i in df[rolling_column]]    
    # rol_ing sum, to get moving (5 mins) latency sum
    df['RS_sum'] = df['sum'].rolling(5).sum()
    # 'len' is number of cars (length of the list Latency Raw)
    df['len'] = [len(i) for i in df[rolling_column]]
    # RS_len is the 'rol_ing sum of length' which is the total number of cars in the past 5 mins  
    df['RS_len'] = df['len'].rolling(5).sum()
    # get the moving average of 5 mins latency from the raw data
    df['MA_latency'] = df['RS_sum']/df['RS_len']
    return df['MA_latency']

# get different past time latency by 5_med_latency
def get_past_thr(thr_mins):
    df_past_thr = data.shift(thr_mins)
    return df_past_thr

# get different past time speed by 5_med_latency
all_latency_lt = []
for i in df.Latency_Raw:
    all_latency_lt += i
df_all_latency = pd.DataFrame(all_latency_lt, columns=['All latency'])

def get_past_speed(thr_mins):
    thr_sec = thr_mins*60
    median_thr = float(df_all_latency[df_all_latency['All latency'] >= thr_sec].median())
    
    # replacing the latency in the raw data list by the mean calculted above
    replaced_latency = []
    for latency_raw_lt in df['Latency_Raw']:
        replaced_latency.append([latency if latency < median_thr else median_thr for latency in latency_raw_lt])
        
    df['Replaced_Latency_Raw'] = replaced_latency
    df_replaced_MA_latency = get_rolling_median_latency(df_latency_raw=df['Replaced_Latency_Raw'])
    
    df_past_thr = df_replaced_MA_latency.shift(thr_mins)
    df_past_speed = 17/df_past_thr*3600  #change distance (etc 17 = 509-376)
    return df_past_speed

