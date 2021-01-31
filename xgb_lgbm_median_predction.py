#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:13:36 2021

@author: zyhhhh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 09:58:16 2020
@ ZHAOYUHENG
"""
# import os,sys
# sys.path.append(os.getcwd())
# import ML_compare
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
 # '376_past_5_median','376_past_10_median','376_past_15_median','376_past_20_median',
 # '413_past_5_median','413_past_10_median','413_past_15_median',
 # '467_past_5_median','467_past_10_median','467_past_15_median',
 '376_413_past5_median_speed_starttime','413_467_past5_median_speed_starttime','467_509_past5_median_speed_starttime',
 '376_509_past10_median_speed_starttime','413_509_past10_median_speed_starttime','467_509_past10_median_speed_starttime',
 '376_509_past15_median_speed_starttime','413_509_past15_median_speed_starttime',
 '376_509_past20_median_speed_starttime']
    
    accuracy = {'lgbm_mse':[],'XGB_mse':[],'lgbm_mape':[],'XGB_mape':[]}
    
    # for n in range(len(weekday)):			#Set range(1) -> range(5) if run from Monday to Friday
    for wi in range(5):

            weekday = [wi]
            df_wd = sep_weekday(df, weekday)
            df_train_or, df_test_or = sep_train_test(df_wd, weekday=weekday, testing_days=testing_days13)			# cahneg [3] to [n]
            df_train_or =  df_train_or.replace([np.inf, -np.inf], np.nan)
            df_test_or =  df_test_or.replace([np.inf, -np.inf], np.nan)
        
            df_train = df_train_or.dropna()
            df_test = df_test_or.dropna()
            ################################
            #predict with medians###########
            ################################
        
            df_train_pred  = pd.DataFrame()
            df_test_pred   = pd.DataFrame()
        
            X_train_new = df_train
            X_test_new = df_test
        
            y_train_new = df_train
            y_test_new = df_test
        
            X_train_new = X_train_new[input_columns_new]
            X_test_new  = X_test_new[input_columns_new]
        
            y_train_new = y_train_new[target_columns]
            y_test_new = y_test_new[target_columns]
            
            
            def mean_absolute_percentage_error(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                return np.mean(np.abs((y_true - y_pred) / y_true))
            
            # import optuna
            # from sklearn.metrics import explained_variance_score
            # from optuna.samplers import TPESampler
            # sampler = TPESampler(seed=666)
            
            # def create_model(trial):
            #     num_leaves = trial.suggest_int("num_leaves", 2, 31)
            #     n_estimators = trial.suggest_int("n_estimators", 20, 300)
            #     max_depth = trial.suggest_int('max_depth', 3, 9)
            #     min_child_samples = trial.suggest_int('min_child_samples', 100, 1200)
            #     learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.99)
            #     min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5, 90)
            #     bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.0001, 1.0)
            #     feature_fraction = trial.suggest_uniform('feature_fraction', 0.0001, 1.0)
            #     model = lgb.LGBMRegressor(
            #         num_leaves=num_leaves,
            #         n_estimators=n_estimators, 
            #         max_depth=max_depth, 
            #         learning_rate=learning_rate
            #     )
            #     return model
            
            # def objective(trial):
            #     model = create_model(trial)
            #     model.fit(X_train_new, y_train_new)
            #     preds = model.predict(X_test_new)
            #     score = explained_variance_score(y_test_new, preds)
            #     return score
        
            # # run optuna
            # study = optuna.create_study(direction="maximize", sampler=sampler)
            # study.optimize(objective, n_trials=5)
            # params = study.best_params    
            
            # params = {'num_leaves': 5, 'n_estimators': 139, 'max_depth': 5, 'min_child_samples': 141, 'learning_rate': 0.5501388733350978, 'min_data_in_leaf': 41, 'bagging_fraction': 0.13871471512702133, 'feature_fraction': 0.6656037894272311}
            
            
            # start_time = time.time()
            # gbm = lgb.LGBMRegressor()
            # gbm.fit(X_train_new, y_train_new,
            #         eval_set=[(X_test_new, y_test_new)],
            #         eval_metric='l1',
            #         early_stopping_rounds=5)
            # # predict
            # y_prediction = gbm.predict(X_test_new, num_iteration=gbm.best_iteration_)
            # df_test['lgbm_pre']=y_prediction
            # # eval
            # print('\n')
            # print('\n prediction by lgbm')
            # print('The rmse of prediction is:', mean_squared_error(df_test['5_med_latency'], y_prediction) ** 0.5)
            # print('The mae of prediction is:', mean_absolute_error(df_test['5_med_latency'], y_prediction))
            # print('The mape of prediction is:', mean_absolute_percentage_error(df_test['5_med_latency'], y_prediction))
            # print('%RMSE_test',mean_squared_error(df_test['5_med_latency'], y_prediction) ** 0.5/df_test['5_med_latency'].mean())
            # print("--- %s seconds ---" % (time.time() - start_time))
            # accuracy['lgbm_mse'].append(mean_squared_error(df_test['5_med_latency'], y_prediction))
            # accuracy['lgbm_mape'].append(mean_absolute_percentage_error(df_test['5_med_latency'], y_prediction))
            
            
            print('\n prediction by xgboost')
            start_time1=time.time()
            pred_model = xgb.XGBRegressor()
            pred_model.fit(X_train_new, y_train_new)
            print('Predicting')
            y_train_pred = pred_model.predict(X_train_new)
            X_train_new['pred_new'] = y_train_pred
        
            #X_test = scaler_X.transform(X_test)
            y_test_pred = pred_model.predict(X_test_new)
            X_test_new['pred_new'] = y_test_pred
        
            df_train_pred  =   X_train_new
            df_test_pred   =   X_test_new
        
            df_train_pred = df_train_pred.sort_index()
            df_test_pred = df_test_pred.sort_index()
        
            df_train['pred_new'] = df_train_pred['pred_new']
            df_test['pred_new']  = df_test_pred['pred_new']
            df0=df_test['pred_new']
            
            MAE_train = mean_absolute_error(df_train['5_med_latency'], df_train['pred_new'])
            MAE_test = mean_absolute_error(df_test['5_med_latency'], df_test['pred_new'])
            MSE_train = mean_squared_error(df_train['5_med_latency'], df_train['pred_new'])
            MSE_test = mean_squared_error(df_test['5_med_latency'], df_test['pred_new'])
            MAPE_train = mean_absolute_percentage_error(df_train['5_med_latency'], df_train['pred_new'])
            MAPE_test = mean_absolute_percentage_error(df_test['5_med_latency'], df_test['pred_new'])
            print('MAE_train:%f,MAE_test:%f'%(MAE_train,MAE_test))
            print('MSE_train:%f,MSE_test:%f'%(MSE_train,MSE_test))
            print('MAPE_train:%f,MAPE_test:%f'%(MAPE_train,MAPE_test))
            print('RMSE_test',MSE_test**0.5)
            mean1 = df_test['5_med_latency'].mean()
            print('%RMSE_test',MSE_test**0.5/mean1)
            print("--- %s seconds ---" % (time.time() - start_time1))
            accuracy['XGB_mse'].append(MSE_test)
            accuracy['XGB_mape'].append(MAPE_test)
            # plot_importance(pred_model,max_num_features=10,grid=False)
            # plt.show()
            