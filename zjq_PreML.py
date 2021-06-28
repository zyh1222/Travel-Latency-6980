# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:58:50 2021

@author: ZJQ
2021/01/19
import this .py and use
X_train_new,X_test_new,y_train_new,y_test_new = preprocessing(picklepath,wi)
to extract training and test set
"""
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
month = 12

testing_days13 = [[datetime.date(2017, 1, 30), datetime.date(2017, 2, 27), datetime.date(2017, 3, 20), datetime.date(2017, 4, 17), datetime.date(2017, 5, 1), datetime.date(2017, 6, 19), datetime.date(2017, 7, 24), datetime.date(2017, 8, 21), datetime.date(2017, 9, 25), datetime.date(2017, 10, 30), datetime.date(2017, 11, 27), datetime.date(2017, 12, 25)],
                  [datetime.date(2017, 1, 24), datetime.date(2017, 2, 14), datetime.date(2017, 3, 21), datetime.date(2017, 4, 11), datetime.date(2017, 5, 9), datetime.date(2017, 6, 27), datetime.date(2017, 7, 25), datetime.date(2017, 8, 15), datetime.date(2017, 9, 5), datetime.date(2017, 10, 10), datetime.date(2017, 11, 7), datetime.date(2017, 12, 5)], 
                  [datetime.date(2017, 1, 11), datetime.date(2017, 2, 1), datetime.date(2017, 3, 15), datetime.date(2017, 4, 5), datetime.date(2017, 5, 17), datetime.date(2017, 6, 21), datetime.date(2017, 7, 5), datetime.date(2017, 8, 16), datetime.date(2017, 9, 13), datetime.date(2017, 10, 11), datetime.date(2017, 11, 29), datetime.date(2017, 12, 27)], 
                  [datetime.date(2017, 1, 5), datetime.date(2017, 2, 2), datetime.date(2017, 3, 30), datetime.date(2017, 4, 20), datetime.date(2017, 5, 11), datetime.date(2017, 6, 22), datetime.date(2017, 7, 20), datetime.date(2017, 8, 17), datetime.date(2017, 9, 28), datetime.date(2017, 10, 5), datetime.date(2017, 11, 16), datetime.date(2017, 12, 21)], 
                  [datetime.date(2017, 1, 20), datetime.date(2017, 2, 10), datetime.date(2017, 3, 3), datetime.date(2017, 4, 7), datetime.date(2017, 5, 5), datetime.date(2017, 6, 30), datetime.date(2017, 7, 7), datetime.date(2017, 8, 11), datetime.date(2017, 9, 8), datetime.date(2017, 10, 13), datetime.date(2017, 11, 24), datetime.date(2017, 12, 29)], 
                  [datetime.date(2017, 1, 28), datetime.date(2017, 2, 11), datetime.date(2017, 3, 18), datetime.date(2017, 4, 1), datetime.date(2017, 5, 6), datetime.date(2017, 6, 17), datetime.date(2017, 7, 8), datetime.date(2017, 8, 26), datetime.date(2017, 9, 23), datetime.date(2017, 10, 7), datetime.date(2017, 11, 4), datetime.date(2017, 12, 23)], 
                  [datetime.date(2017, 1, 1), datetime.date(2017, 2, 19), datetime.date(2017, 3, 5), datetime.date(2017, 4, 9), datetime.date(2017, 5, 14), datetime.date(2017, 6, 4), datetime.date(2017, 7, 9), datetime.date(2017, 8, 13), datetime.date(2017, 9, 10), datetime.date(2017, 10, 22), datetime.date(2017, 11, 5), datetime.date(2017, 12, 31)]]
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

input_columns_old = ['339_in', '339_out', '339_acc', '339_en', '339_ex', '376_in', '376_out',
 '376_acc', '376_en', '376_ex', '413_in', '413_out', '413_acc', '413_en',
 '413_ex', '467_in', '467_out', '467_acc', '467_en', '467_ex', '339_in_p1',
 '339_out_p1', '339_acc_p1', '339_en_p1', '339_ex_p1', '376_in_p1',
 '376_out_p1', '376_acc_p1', '376_en_p1', '376_ex_p1', '413_in_p1',
 '413_out_p1', '413_acc_p1', '413_en_p1', '413_ex_p1', '467_in_p1',
 '467_out_p1', '467_acc_p1', '467_en_p1', '467_ex_p1', '339_in_p2',
 '339_out_p2', '339_acc_p2', '339_en_p2', '339_ex_p2', '376_in_p2',
 '376_out_p2', '376_acc_p2', '376_en_p2', '376_ex_p2', '413_in_p2',
 '413_out_p2', '413_acc_p2', '413_en_p2', '413_ex_p2', '467_in_p2',
 '467_out_p2', '467_acc_p2', '467_en_p2', '467_ex_p2', '339_in_p3',
 '339_out_p3', '339_acc_p3', '339_en_p3', '339_ex_p3', '376_in_p3',
 '376_out_p3', '376_acc_p3', '376_en_p3', '376_ex_p3', '413_in_p3',
 '413_out_p3', '413_acc_p3', '413_en_p3', '413_ex_p3', '467_in_p3',
 '467_out_p3', '467_acc_p3', '467_en_p3', '467_ex_p3', '339_in_p4',
 '339_out_p4', '339_acc_p4', '339_en_p4', '339_ex_p4', '376_in_p4',
 '376_out_p4', '376_acc_p4', '376_en_p4', '376_ex_p4', '413_in_p4',
 '413_out_p4', '413_acc_p4', '413_en_p4', '413_ex_p4', '467_in_p4',
 '467_out_p4', '467_acc_p4', '467_en_p4', '467_ex_p4', '339_in_p5',
 '339_out_p5', '339_acc_p5', '339_en_p5', '339_ex_p5', '376_in_p5',
 '376_out_p5', '376_acc_p5', '376_en_p5', '376_ex_p5', '413_in_p5',
 '413_out_p5', '413_acc_p5', '413_en_p5', '413_ex_p5', '467_in_p5',
 '467_out_p5', '467_acc_p5', '467_en_p5', '467_ex_p5', 'ex_tot', 'en_tot',
 'acc_tot', 'past15_median_speed_starttime', 'past20_median_speed_starttime', 'past30_median_speed_starttime',
 'past40_median_speed_starttime','p_l_median','5_med_latency_arrival',
 # '376_past_5_median','376_past_10_median','376_past_15_median','376_past_20_median',
 # '413_past_5_median','413_past_10_median','413_past_15_median',
 # '467_past_5_median','467_past_10_median','467_past_15_median',
 '376_413_past5_median_speed_starttime','413_467_past5_median_speed_starttime','467_509_past5_median_speed_starttime',
 '376_509_past10_median_speed_starttime','413_509_past10_median_speed_starttime','467_509_past10_median_speed_starttime',
 '376_509_past15_median_speed_starttime','413_509_past15_median_speed_starttime',
 '376_509_past20_median_speed_starttime']

input_columns_new = ['339_in','past15_median_speed_starttime','p_l_median',
 '467_509_past5_median_speed_starttime',
 '339_acc',
 '376_509_past10_median_speed_starttime',
 '376_413_past5_median_speed_starttime',
 '339_out',
 'past40_median_speed_starttime',
 'acc_tot',
 '376_acc',
 'past30_median_speed_starttime',
 '413_467_past5_median_speed_starttime',
 '467_acc',
 '413_acc',
 '413_509_past10_median_speed_starttime',
 '376_509_past20_median_speed_starttime']

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
 
def divide_train_testsets_by_date(df,date1,date2):
   # testset_size = 1020
    testset_size = 2040
    df_set = df[df.index.date<date2]
    df_set = df_set[date1<=df_set.index.date]
    print(len(df_set))
    scaler_paras_y = (np.mean(df_set[target_columns]),np.std(df_set[target_columns]))
    scaler = preprocessing.StandardScaler().fit(df_set)
    df_set_scaled = scaler.transform(df_set)
    res = [df_set_scaled[:-testset_size,:-1],df_set_scaled[:-testset_size,-1]]
    res += [df_set_scaled[-testset_size:,:-1],df_set_scaled[-testset_size:,-1:]]
    res += [scaler_paras_y,df_set.index[-testset_size:]]
    #res = [train_x,train_y,test_x,test_y,scaler_params,test_index]
    return res

def lstm_preprocessing(picklepath,wi):
    weekday = [wi] 
    df = pd.read_pickle(picklepath)
    df = df.loc[df.index.year == 2017]
    df = df.loc[df.index.month <= month]
    df = df[input_columns_lstm+target_columns]
    #exclude nan days; 349days left
    for di in nan_days:
        df = df[df.index.date!=di]
    print('wholeset:',len(df))
    df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print('wholeset:',len(df))
    # select weekday
    df_wd = sep_weekday(df, weekday)
    df = df[df[target_columns]>0]
    #df_wd = df_wd[input_columns_lstm+target_columns]
    print('wholeset:',len(df_wd))
    #训练集：前三个月
    date1 =  datetime.date(2017,1,1)
    date2 =  datetime.date(2017,11,15)
    # date3 =  datetime.date(2017,9,7)
    # date4 =  datetime.date(2017,12,31)
    return divide_train_testsets_by_date(df_wd,date1,date2)#,divide_train_testsets_by_date(df_wd,date2,date3),divide_train_testsets_by_date(df_wd,date3,date4)
    
    
def sep_weekday(input_df, weekday0=[]):
    df_wd_all = pd.DataFrame()
    
    for day in weekday0:
    
        df_wd0 = input_df.loc[input_df.index.weekday == day]
        df_wd_all = pd.concat([df_wd_all, df_wd0])
    return df_wd_all

def sep_train_test(input_df, weekday, testing_days):
#   dropping the testing days from the input_df
    df_train = input_df.copy()
    
    testing_days_lt = []
    for i in weekday:
        testing_days_lt += testing_days[i]
        
    for i in testing_days_lt:
        df_train = df_train[df_train.index.date != i]
        
#   Combining the testing days into a dataframe
    df_test = pd.DataFrame()
    for i in testing_days_lt:
        df_test = pd.concat([df_test, input_df[input_df.index.date == i]])
        
    return df_train, df_test

def sep_train_test_random(input_df, weekday, proportion_testing_days=0.2):
#   dropping the testing days from the input_df
    df_train = input_df.copy()
    
    testing_days_lt = []
    for i in weekday:
        testing_days_lt += testing_days[i]
        
    for i in testing_days_lt:
        df_train = df_train[df_train.index.date != i]
        
#   Combining the testing days into a dataframe
    df_test = pd.DataFrame()
    for i in testing_days_lt:
        df_test = pd.concat([df_test, input_df[input_df.index.date == i]])
        
    return df_train, df_test


def xgb_preprocessing(picklepath,wi):
    weekday = [wi] 
    df = pd.read_pickle(picklepath)
    df = df.loc[df.index.year == 2017]
    df = df.loc[df.index.month <= month]
    
    #提取特定weekday
    df_wd = sep_weekday(df, weekday)
    
    df_train_or, df_test_or = sep_train_test(df_wd, weekday=weekday, testing_days=testing_days13)			# cahneg [3] to [n]
    df_train_or =  df_train_or.replace([np.inf, -np.inf], np.nan)
    df_test_or =  df_test_or.replace([np.inf, -np.inf], np.nan)
    
    df_train = df_train_or.dropna()
    df_test = df_test_or.dropna()
    ################################
    #predict with medians###########
    ################################


    X_train_new = df_train
    X_test_new = df_test
    
    y_train_new = df_train
    y_test_new = df_test
    
    X_train_new = X_train_new[input_columns_new]
    X_test_new  = X_test_new[input_columns_new]
    
    y_train_new = y_train_new[target_columns]
    y_test_new = y_test_new[target_columns]
    
    return X_train_new,X_test_new,y_train_new,y_test_new

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))