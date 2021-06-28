# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 00:16:05 2021

@author: ZJQ
"""
import os,sys
sys.path.append(os.getcwd())
from zjq_PreML import *
import time
import datetime
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

import xgboost as xgb
import json

thresholds = [820,800,820,820,850]
#thresholds = [820,800,820,820,820]
#wi = 4
#threshold = thresholds[wi]
picklepath = '/Users/zhangjiaqi1/Public/course/6980/df_median_speed_starttime.pickle'



def get_prediction_xgb(dataset_k1,wi):
    threshold = thresholds[wi]
    #df = pd.read_pickle(picklepath)
    #for wi in range(3,4):
    #wi = 0#0-4：Mon-Fri
    #k = 3#3-fold-cross-validation
    k = 1
    ################
    #2.prepressing##
    ################
    #dataset_k1,dataset_k2,dataset_k3 = lstm_preprocessing(picklepath,wi)
    #dataset_k1 = lstm_preprocessing(picklepath,wi)
    dataset = [dataset_k1]
    l_mse = []
    l_mape = []
    for ki in range(k):
        data = dataset[ki]
        dim = data[0].shape[-1]
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
        
        res = [y_test_rescale,pred_test_rescale,
               y_test_rescale[y_test_rescale<threshold],pred_test_rescale[y_test_rescale<threshold],
               y_test_rescale[y_test_rescale>=threshold],pred_test_rescale[y_test_rescale>=threshold]]
        return res

def get_prediction_xgb2(dataset_k1,wi):
    threshold = thresholds[wi]
    #wi = 0#0-4：Mon-Fri
    k = 1
    ################
    #2.prepressing##
    ################
    #dataset_k1 = lstm_preprocessing(picklepath,wi)
    data = dataset_k1
    l_mse = []
    l_mape = []
    dim = data[0].shape[-1]
    train_x0 =data[0]
    train_y0 = data[1]
    test_x0 =data[2]
    test_y0 = data[3]
    test_y00 = test_y0.reshape(-1,)
    u,std = data[4][0],data[4][1]
    threshold_normalized = ((threshold - u)/std)[0]
    ##############
    train_y = train_y0[train_y0 <threshold_normalized]
    train_x = train_x0[train_y0 < threshold_normalized]
    test_y = test_y0[test_y00 < threshold_normalized]
    test_x = test_x0[test_y00 <threshold_normalized]
    
    
    pred_model = xgb.XGBRegressor()
    pred_model.fit(train_x,train_y)
    pred_test = pred_model.predict(test_x)
    pred_train = pred_model.predict(train_x)
    
    std = data[4][1].values[0]
    u = data[4][0].values[0]
    #rescale the normalized data
    y_train_rescale = train_y.reshape(1,-1)[0]*std+u
    pred_train_rescale = pred_train*std+u
    y_test_rescale1 = test_y.reshape(1,-1)[0]*std+u
    pred_test_rescale1 = pred_test*std+u
    ##########################
    train_y = train_y0[train_y0 >= threshold_normalized]
    train_x = train_x0[train_y0 >= threshold_normalized]
    test_y = test_y0[test_y00 >= threshold_normalized]
    test_x = test_x0[test_y00 >=threshold_normalized]
    
    
    pred_model = xgb.XGBRegressor()
    pred_model.fit(train_x,train_y)
    pred_test = pred_model.predict(test_x)
    pred_train = pred_model.predict(train_x)
    
    std = data[4][1].values[0]
    u = data[4][0].values[0]
    #rescale the normalized data
    y_train_rescale = train_y.reshape(1,-1)[0]*std+u
    pred_train_rescale = pred_train*std+u
    y_test_rescale2 = test_y.reshape(1,-1)[0]*std+u
    pred_test_rescale2 = pred_test*std+u
    
    y_test_rescale0 = np.hstack([y_test_rescale1,y_test_rescale2])
    pred_test_rescale0 = np.hstack([pred_test_rescale1,pred_test_rescale2])
    
    res = [y_test_rescale0,pred_test_rescale0,
           y_test_rescale1,pred_test_rescale1,
           y_test_rescale2,pred_test_rescale2]
    return res

def get_prediction_past15(dataset_k1,wi):
    threshold = thresholds[wi]
    k = 1
    ################
    #2.prepressing##
    ################
    #dataset_k1 = lstm_preprocessing(picklepath,wi)
    timeindex = dataset_k1[-1]
    df = pd.read_pickle(picklepath)
    testset = df.loc[timeindex,:]
    y_test_rescale = testset['5_med_latency']
    y1__ = testset['past_15_med']
    
    res = [y_test_rescale,y1__,
           y_test_rescale[y_test_rescale<threshold],y1__[y_test_rescale<threshold],
            y_test_rescale[y_test_rescale>=threshold],y1__[y_test_rescale>=threshold]]
    return res
    
def get_mape(res):
    m1 = mean_absolute_percentage_error(res[0],res[1])
    m2 = mean_absolute_percentage_error(res[2],res[3])
    m3 = mean_absolute_percentage_error(res[4],res[5])
    return [m1,m2,m3]

def get_mse(res):
    m1 = mean_squared_error(res[0],res[1])
    m2 = mean_squared_error(res[2],res[3])
    m3 = mean_squared_error(res[4],res[5])
    return [m1,m2,m3]

def list_to_dict(l):
    #print("yes")
    res_dict = {"model":None,
                "weekday":None,
                "Fold":None,
                "test_mse_overall":None,
                "test_mse_freeflow":None,
                "test_mse_congestion":None,
                "test_mape_overall":None,
                "test_mape_freeflow":None,
                "test_mape_congestion":None}
    
    for ki,vi in zip(res_dict.keys(),l):
        res_dict[ki] = vi

    f = open("/Users/zhangjiaqi1/Public/course/6980/Git/Travel-Latency-6980/ressults_4_27/"+l[0]+"_"+str(l[1])+"_k"+str(l[2])+".json","w")
    f.write(json.dumps(res_dict))
    f.close()


#if __name__ == '__main__':
def main_multimodel(dataset_k1,wi,fold_i):
        
        res1 = get_prediction_xgb(dataset_k1,wi)
        res2 = get_prediction_xgb2(dataset_k1,wi)
        res3 = get_prediction_past15(dataset_k1,wi)

        #得到mse
        mse1 = get_mse(res1)
        mse2 = get_mse(res2)
        mse3 = get_mse(res3)
        #画mse直方图
        
        #画mape直方图
        mape1 = get_mape(res1)
        mape2 = get_mape(res2)
        mape3 = get_mape(res3)
        
        #write results into json files
        
        
        model = "xgb1"
        res_all = list_to_dict([model,wi,fold_i]+mse1+mape1)
        model = "xgb2"
        res_all = list_to_dict([model,wi,fold_i]+mse2+mape2)
        model = "past15"
        res_all = list_to_dict([model,wi,fold_i]+mse3+mape3)
        
        
        
        
        
        
        