#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:00:36 2021

@author: zhangjiaqi1
"""
import numpy as np
import pandas as pd
import datetime
import random
import sys
import os
sys.path.append(os.getcwd())
from zjq_multimodel_kfcv import *

wi = 3

random.seed(6980)
picklepath = '/Users/zhangjiaqi1/Public/course/6980/df_median_speed_starttime.pickle'
df0 = pd.read_pickle(picklepath)
k = 10

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
month = 12
nan_days = [datetime.date(2017, 1, 4),
            datetime.date(2017, 1, 19),
            datetime.date(2017, 1, 25),
            datetime.date(2017, 1, 26),
            datetime.date(2017, 2, 21),
            datetime.date(2017, 5, 22),
            datetime.date(2017, 5, 23),
            datetime.date(2017, 5, 24),
            datetime.date(2017, 5, 25),
            datetime.date(2017, 5, 26),
            datetime.date(2017, 5, 29),
            datetime.date(2017, 5, 30),
            datetime.date(2017, 5, 31),
            datetime.date(2017, 8, 30),
            datetime.date(2017, 8, 31),
            datetime.date(2017, 10, 19),#Dec.21st
             ]
target_columns = ['5_med_latency']
input_columns_lstm = ['5_med_latency_arrival',
                      'p_l_median',
                      #'past15_median_speed_starttime',
                       # 'past40_median_speed_starttime',
                       # 'past30_median_speed_starttime',
                      'past_15_med',
                      'past_30_med',
                      'past_40_med',
                        '376_509_past10_median_speed_starttime',
                         '376_413_past5_median_speed_starttime',
                         '467_509_past5_median_speed_starttime',
                         '413_509_past10_median_speed_starttime',
                         '376_509_past20_median_speed_starttime',
                         '413_467_past5_median_speed_starttime',
 '339_in',
 '339_acc',

 '339_out',
 'acc_tot',
 '376_acc',

 '467_acc',
 '413_acc']
 


df = df0[df0.index.weekday == wi]

df = df.loc[df.index.year == 2017]
df = df.loc[df.index.month <= month]
df = df[input_columns_lstm+target_columns]
df = df[(df[target_columns]>0).values]
#exclude nan days; 349days left
for di in nan_days:
    df = df[df.index.date!=di]
df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
print('wholeset:',len(df))

scaler_params_y = (np.mean(df[target_columns]),np.std(df[target_columns]))


df["date"] = df.index.date
#df = df.sample(frac=1)

#shuffle by date
gp = df.groupby("date")
Agp = list(gp)
random.shuffle(Agp)
groups_df = [_df[1] for _df in Agp]
df = pd.concat(groups_df)

del df ['date'] 

#Normalize
scaler = preprocessing.StandardScaler().fit(df)
df_scaled = scaler.transform(df)
L = len(df)

#for i in range(k):


#sparse

def run_fold_i(i): #the i-th fold
#i = 0
    #start_index,end_index = i*L//k,(i+1)*L//k
    #5 days for testset
    start_index,end_index = 4*i*1020,4*(i+1)*1020
    test_set = df_scaled[start_index:end_index,:]
    train_set = np.vstack([df_scaled[:start_index,:],df_scaled[end_index:,:]])
    
    train_x = train_set[:,:-1]
    train_y = train_set[:,-1]
    test_x = test_set[:,:-1]
    test_y = test_set[:,-1]
    test_index = df.index[start_index:end_index]
    
    res = [train_x,train_y,test_x,test_y,scaler_params_y,test_index]
    main_multimodel(res,wi,i)


import multiprocessing

if __name__ == '__main__':
    process_num = 5
    pool = multiprocessing.Pool(processes=process_num)
    fold_nums = list(range(k))
    
    pool.map(run_fold_i, fold_nums)

    pool.close()
    pool.join()
    print("End...........")
    # for i in fold_nums:
    #     run_fold_i(i)



