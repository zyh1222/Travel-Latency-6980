#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:34:43 2021

@author: zyhhhh
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn import mixture
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import datetime
import random
import time
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))
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

if __name__=='__main__':
    
    picklepath = '/Users/zhaoyuheng/Downloads/OneDrive_1_6-15-2020/Data/df_median_speed_starttime.pickle'
    testing_days13 = [[datetime.date(2017, 1, 30), datetime.date(2017, 2, 27), datetime.date(2017, 3, 20), datetime.date(2017, 4, 17), datetime.date(2017, 5, 1), datetime.date(2017, 6, 19), datetime.date(2017, 7, 24), datetime.date(2017, 8, 21), datetime.date(2017, 9, 25), datetime.date(2017, 10, 30), datetime.date(2017, 11, 27), datetime.date(2017, 12, 25)], [datetime.date(2017, 1, 24), datetime.date(2017, 2, 14), datetime.date(2017, 3, 21), datetime.date(2017, 4, 11), datetime.date(2017, 5, 9), datetime.date(2017, 6, 27), datetime.date(2017, 7, 25), datetime.date(2017, 8, 15), datetime.date(2017, 9, 5), datetime.date(2017, 10, 10), datetime.date(2017, 11, 7), datetime.date(2017, 12, 5)], [datetime.date(2017, 1, 11), datetime.date(2017, 2, 1), datetime.date(2017, 3, 15), datetime.date(2017, 4, 5), datetime.date(2017, 5, 17), datetime.date(2017, 6, 21), datetime.date(2017, 7, 5), datetime.date(2017, 8, 16), datetime.date(2017, 9, 13), datetime.date(2017, 10, 11), datetime.date(2017, 11, 29), datetime.date(2017, 12, 27)], [datetime.date(2017, 1, 5), datetime.date(2017, 2, 2), datetime.date(2017, 3, 30), datetime.date(2017, 4, 20), datetime.date(2017, 5, 11), datetime.date(2017, 6, 22), datetime.date(2017, 7, 20), datetime.date(2017, 8, 17), datetime.date(2017, 9, 28), datetime.date(2017, 10, 5), datetime.date(2017, 11, 16), datetime.date(2017, 12, 21)], [datetime.date(2017, 1, 20), datetime.date(2017, 2, 10), datetime.date(2017, 3, 3), datetime.date(2017, 4, 7), datetime.date(2017, 5, 5), datetime.date(2017, 6, 30), datetime.date(2017, 7, 7), datetime.date(2017, 8, 11), datetime.date(2017, 9, 8), datetime.date(2017, 10, 13), datetime.date(2017, 11, 24), datetime.date(2017, 12, 29)], [datetime.date(2017, 1, 28), datetime.date(2017, 2, 11), datetime.date(2017, 3, 18), datetime.date(2017, 4, 1), datetime.date(2017, 5, 6), datetime.date(2017, 6, 17), datetime.date(2017, 7, 8), datetime.date(2017, 8, 26), datetime.date(2017, 9, 23), datetime.date(2017, 10, 7), datetime.date(2017, 11, 4), datetime.date(2017, 12, 23)], [datetime.date(2017, 1, 1), datetime.date(2017, 2, 19), datetime.date(2017, 3, 5), datetime.date(2017, 4, 9), datetime.date(2017, 5, 14), datetime.date(2017, 6, 4), datetime.date(2017, 7, 9), datetime.date(2017, 8, 13), datetime.date(2017, 9, 10), datetime.date(2017, 10, 22), datetime.date(2017, 11, 5), datetime.date(2017, 12, 31)]]
    month = 12
    # weekday = [3]
    df = pd.read_pickle(picklepath)
    df = df.loc[df.index.year == 2017]
    df = df.loc[df.index.month <= month]
    # df=df[df['5_med_latency_arrival']>0]

    target_columns = ['5_med_latency']
    input_columns_new = ['339_in', '339_out', '339_acc', '339_en', '339_ex', '376_in', '376_out',
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
 '376_413_past5_median_speed_starttime','413_467_past5_median_speed_starttime','467_509_past5_median_speed_starttime',
 '376_509_past10_median_speed_starttime','413_509_past10_median_speed_starttime','467_509_past10_median_speed_starttime',
 '376_509_past15_median_speed_starttime','413_509_past15_median_speed_starttime',
 '376_509_past20_median_speed_starttime']
    accuracy = {'fd_mse':[],'fd_mape':[]}

    def load(f):
        path = '../Data/'+f+'.pickle'
        p = pd.read_pickle(path)
        print(f + ' loaded')
        return p
    
    
    fl = ['L201701.pickle',
          'L201704.pickle',
          'L201707.pickle',
          'L201710.pickle']

    flux = []
    acc = []
    for f in fl:
        fn = f.split('.')
        if fn[1] == 'pickle':
            di = load(fn[0])
            flux.append(di['Out_flux_TS'][:-1440])
            acc.append(di['Accumulation_TS'][:-1440])
    
    flux = pd.concat(flux)['01F0339S']['01F0509S']
    acc = pd.concat(acc)['01F0339S']['01F0509S']
    
    fd = pd.DataFrame({'acc': acc.values, 'flux':flux.values}, index=flux.index)
    fd = fd[(fd.index.hour >= 6) & (fd.index.hour < 23)]    # We drop the mid-night data, only use the data from 6 a.m. - 11 p.m.
    fd2017 = fd[fd.index.year == 2017]    # Only using 2017 data
    fd2018 = pd.read_pickle('../Data/fd_2018.pickle')
    fd = fd.append(fd2018)
    
    df[['acc', 'flux']] = fd
    eqns = {0: {'m_0':0.074465368, 'c_0':-16.81868553, 'ver_line_0':448.2378141, 'lat_b':1431.82909}, 
            1: {'m_0':0.095479372, 'c_0':-26.58672537, 'ver_line_0':461.4185264, 'lat_b':1378.892919}, 
            2: {'m_0':0.082656165, 'c_0':-17.62928122, 'ver_line_0':373.4126797, 'lat_b':1365.006988}, 
            3: {'m_0':0.09605792, 'c_0':-28.51470362, 'ver_line_0':477.4002431, 'lat_b':1374.795257}, 
            4: {'m_0':0.098130292, 'c_0':-33.04279938, 'ver_line_0':514.3545896, 'lat_b':1423.561222}, 
            5: {'m_0':0.11671963, 'c_0':-36.76775095, 'ver_line_0':457.9361863, 'lat_b':1113.927746}, 
            6: {'m_0':0.367354657, 'c_0':-40.51262128, 'ver_line_0':124.8464519, 'lat_b':1044.010424}}
    FD_v = []
    for wi in range(5):		# Change to 5 if run from Monday to Fridaty
     	weekday = [wi]
     	df_wd = sep_weekday(df, [wi])
     	cond = []
     	for i in range(len(df_wd)):
     	    outflux_i = df_wd['flux'][i]
     	    accumulation_i = df_wd['acc'][i]
    
     	#             if smaller than the vertical line, then free
     	    if accumulation_i <= eqns[wi]['ver_line_0']:
     	        cond.append('free')
     	#             if the above the inclined line, then free
     	    elif outflux_i >= eqns[wi]['m_0']*accumulation_i + eqns[wi]['c_0']:
     	        cond.append('free')
     	#             otherwise is congested
     	    else: 
     	        cond.append('con')
    
     	df_wd['by_fd'] = cond   
    
    
     	df_train_or, df_test_or = sep_train_test(df_wd, weekday=weekday, testing_days=testing_days13)
    
    
     	df_train_or =  df_train_or.replace([np.inf, -np.inf], np.nan)
     	df_test_or =  df_test_or.replace([np.inf, -np.inf], np.nan)
    
     	df_train = df_train_or.dropna()
     	df_test = df_test_or.dropna()
    
     	######## Testing
    
     	#######################################################
     	df_train_pred  = pd.DataFrame()
     	df_test_pred   = pd.DataFrame()
    
     	##############################################################
     	X_train_new = df_train[df_train['by_fd'] == 'free']
     	X_test_new = df_test[df_test['by_fd'] == 'free']
    
     	y_train_new = df_train[df_train['by_fd'] == 'free']
     	y_test_new = df_test[df_test['by_fd'] == 'free']
    
     	X_train_new = X_train_new[input_columns_new]
     	X_test_new  = X_test_new[input_columns_new]
    
     	y_train_new = y_train_new[target_columns]
     	y_test_new = y_test_new[target_columns]
    
    
     	pred_model = xgb.XGBRegressor()
     	pred_model.fit(X_train_new, y_train_new)
     	y_train_pred = pred_model.predict(X_train_new)
     	X_train_new['pred_new'] = y_train_pred
    
     	#X_test = scaler_X.transform(X_test)
     	y_test_pred = pred_model.predict(X_test_new)
     	X_test_new['pred_new'] = y_test_pred
    
     	df_train_pred  =   X_train_new
     	df_test_pred   =   X_test_new
    
     	#############################################################
    
     	X_train_new = df_train[df_train['by_fd'] == 'con']
     	X_test_new = df_test[df_test['by_fd'] == 'con']
    
     	y_train_new = df_train[df_train['by_fd'] == 'con']
     	y_test_new = df_test[df_test['by_fd'] == 'con']
    
     	X_train_new = X_train_new[input_columns_new]
     	X_test_new= X_test_new[input_columns_new]
    
     	y_train_new = y_train_new[target_columns]
     	y_test_new = y_test_new[target_columns]
    
    
     	pred_model = xgb.XGBRegressor()
     	pred_model.fit(X_train_new, y_train_new)
     	y_train_pred = pred_model.predict(X_train_new)
     	X_train_new['pred_new'] = y_train_pred
    
     	#X_test = scaler_X.transform(X_test)
     	y_test_pred = pred_model.predict(X_test_new)
     	X_test_new['pred_new'] = y_test_pred
    
     	df_train_pred = pd.concat([df_train_pred, X_train_new])
     	df_test_pred  = pd.concat([df_test_pred, X_test_new])
    
     	######################################################################
     	df_train_pred = df_train_pred.sort_index()
     	df_test_pred = df_test_pred.sort_index()
    
     	df_train['pred_new'] = df_train_pred['pred_new']
     	df_test['pred_new']  = df_test_pred['pred_new']
    
     	MSE_train = mean_squared_error(df_train['5_med_latency'], df_train['pred_new'])
     	MSE_test = mean_squared_error(df_test['5_med_latency'], df_test['pred_new'])
     	MAPE_test = mean_absolute_percentage_error(df_test['5_med_latency'], df_test['pred_new'])
     	print('MSE_train:%f,MSE_test:%f'%(MSE_train,MSE_test))
     	print('MAPE_test:%f',MAPE_test)
     	accuracy['fd_mse'].append(MSE_test)
     	accuracy['fd_mape'].append(MAPE_test)
    pd.DataFrame(accuracy).T.to_csv('/Users/zhaoyuheng/Downloads/OneDrive_1_6-15-2020/Data/acc_results1.csv',mode='a')
    