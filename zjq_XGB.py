# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 00:16:05 2021

@author: ZJQ
"""
import os,sys
sys.path.append("./data/Git/Travel-Latency-6980")
from zjq_PreML import *
import time
import datetime
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

import xgboost as xgb

if __name__ == '__main__':
    picklepath = './data/df_median_speed_starttime.pickle'
    save_path = './results'
    #df = pd.read_pickle(picklepath)
    #for wi in range(3,4):
    wi = 1#0-4：Mon-Fri
    #k = 3#3-fold-cross-validation
    k = 0
    ################
    #2.prepressing##
    ################
    #dataset_k1,dataset_k2,dataset_k3 = lstm_preprocessing(picklepath,wi)
    dataset_k1 = lstm_preprocessing(picklepath,wi)
    #test: tues: 11/21 11/28
    dataset = [dataset_k1]
    l_mse = []
    l_mape = []
    #for ki in range(k):
    ki = 0
    if True:
        data = dataset[ki]
        dim = len(data[0][0])
        train_x =data[0]
        train_y = data[1]
        test_x =data[2]
        test_y = data[3]
        pred_model = xgb.XGBRegressor()
        pred_model.fit(train_x,train_y)
        pred_test = pred_model.predict(test_x)
        pred_train = pred_model.predict(train_x)
        
        std = data[4][1].values[0]
        u = data[4][0].values[0]
        #rescale the normalized data
        y_train_rescale = data[1].reshape(1,-1)[0]*std+u
        pred_train_rescale = pred_train*std+u
        y_test_rescale = data[3].reshape(1,-1)[0]*std+u
        pred_test_rescale = pred_test*std+u
        l_mse.append((mean_squared_error(y_train_rescale,pred_train_rescale),mean_squared_error(y_test_rescale,pred_test_rescale)))
        l_mape.append((mean_absolute_percentage_error(y_train_rescale,pred_train_rescale),mean_absolute_percentage_error(y_test_rescale,pred_test_rescale)))
        df = pd.read_pickle(picklepath)
        #plot all predicton
        lenth = 1020
        
        time_series = np.arange(1,lenth+1)
        
        timeindex = dataset_k1[-1]
        testset = df.loc[timeindex,:]
        y_past15 = testset['past_15_med'].values
        plt.figure(figsize=[20,10])
        plt.plot(time_series,y_test_rescale[:lenth],color = "b",label = "GroundTruth")
        plt.plot(time_series,y_past15[:lenth],color = "g",label = "baseline:past15")
        plt.plot(time_series,pred_test_rescale[:lenth],color = "r",label = "Prediction")
        plt.ylabel("Travel Time (s)",fontsize = 18)
        plt.xlabel("Time of a day",fontsize = 18)
        hours = np.arange(0,lenth+1,60)
        hours_time = [str(i)+':00' for i in range(6,24)]
        plt.xticks(hours, hours_time)
        plt.title("XGBoost Prediction on 2017/11/21, Tuesday",fontsize = 18)
        plt.legend(fontsize = 18)


        plt.figure()
        plt.plot(data[3].reshape(1,-1)[0],label='real')
        plt.plot(pred_test,label='pred')
        plt.legend()
                #save prediction data
        test_data = np.vstack([pred_test_rescale,y_test_rescale])
        df_test_data = pd.DataFrame(test_data.T,columns = ['y_pred','y_true'])
        
        df_test_data.to_csv(os.path.join(save_path,'pred_3.14_xgb.csv'))
        
        n1 = len(y_test_rescale[ y_test_rescale<950])
        n2 = 2040 - n1
        y1 = y_test_rescale[y_test_rescale<950]
        y1_ = pred_test_rescale[y_test_rescale<950]
        mean_absolute_percentage_error(y1,y1_)
        mse1 = mean_squared_error(y1,y1_)
        
        y2 = y_test_rescale[y_test_rescale>=950]
        y2_ = pred_test_rescale[y_test_rescale>=950]
        mean_absolute_percentage_error(y2,y2_)
        mse2 = mean_squared_error(y2,y2_)
        (mse1 * n1 + mse2*n2)/(n1+n2)
        
        plt.figure()
        plt.plot(y1,label='real')
        plt.plot(y1_,label='pred')
        plt.legend()
        plt.title('Predicted and true latency < Threshold')
        
        plt.figure()
        plt.plot(y2,label='real')
        plt.plot(y2_,label='pred')
        plt.legend()
        plt.title('Predicted and true latency >= Threshold')
        
        timeindex = dataset_k1[-1]
        testset = df.loc[timeindex,:]
        
        y1__ = testset['past_15_med']
        y1__ = y1__[y_test_rescale<950]
        mean_absolute_percentage_error(y1,y1__)
        
        y2__ = testset['past_15_med']
        y2__ = y2__[y_test_rescale>=950]
        mean_absolute_percentage_error(y2,y2__)
        
        plt.figure()
        plt.plot(y1,label='real')
        plt.plot(y1__.values,label='past15')
        plt.legend()
        plt.title('Predicted and true latency < Threshold')
        
        plt.figure()
        plt.plot(y2,label='real')
        plt.plot(y2__.values,label='past15')
        plt.legend()
        plt.title('Predicted and true latency >= Threshold')
        
        plt.figure()
        plt.plot(y_test_rescale,label = 'real_latency')
        plt.plot([950 for i in y_test_rescale],label = '950')
        plt.plot([900 for i in y_test_rescale],label = '900')
        plt.plot([700 for i in y_test_rescale],label = '700')
        plt.legend()
        plt.ylabel('latency')
        