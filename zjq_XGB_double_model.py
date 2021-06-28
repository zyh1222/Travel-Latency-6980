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
thresholds = [820,800,820,820,850]
wi = 4
threshold = thresholds[wi]
picklepath = '/Users/zhangjiaqi1/Public/course/6980/df_median_speed_starttime.pickle'

def get_prediction_xgb():
    
    #picklepath = 
    #df = pd.read_pickle(picklepath)
    #for wi in range(3,4):
    #wi = 0#0-4：Mon-Fri
    #k = 3#3-fold-cross-validation
    k = 1
    ################
    #2.prepressing##
    ################
    #dataset_k1,dataset_k2,dataset_k3 = lstm_preprocessing(picklepath,wi)
    dataset_k1 = lstm_preprocessing(picklepath,wi)
    dataset = [dataset_k1]
    l_mse = []
    l_mape = []
    for ki in range(k):
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
        
        res = [y_test_rescale,pred_test_rescale,
               y_test_rescale[y_test_rescale<threshold],pred_test_rescale[y_test_rescale<threshold],
               y_test_rescale[y_test_rescale>=threshold],pred_test_rescale[y_test_rescale>=threshold]]
        return res

def get_prediction_xgb2():
    
    def re_sort(y_test_rescale1,y_test_rescale2,timeindex_1,timeindex_2):
        df1 = pd.DataFrame(index = timeindex_1)
        df1["latency"] = y_test_rescale1
        df2 = pd.DataFrame(index = timeindex_2)
        df2["latency"] = y_test_rescale2
        df0 = df1.append(df2)
        df0 = df0.sort_index(axis=0)
        return df0.values
    #wi = 0#0-4：Mon-Fri
    k = 1
    ################
    #2.prepressing##
    ################
    dataset_k1 = lstm_preprocessing(picklepath,wi)
    data = dataset_k1
    l_mse = []
    l_mape = []
    dim = len(data[0][0])
    train_x0 =data[0]
    train_y0 = data[1]
    test_x0 =data[2]
    test_y0 = data[3]
    test_y00 = test_y0.reshape(-1,)
    u,std = data[4][0],data[4][1]
    threshold_normalized = ((threshold - u)/std)[0]
    
    timeindex = data[5]
    ##############
    train_y = train_y0[train_y0 <threshold_normalized]
    train_x = train_x0[train_y0 < threshold_normalized]
    test_y = test_y0[test_y00 < threshold_normalized]
    test_x = test_x0[test_y00 <threshold_normalized]
    timeindex_1 = timeindex[test_y00 <threshold_normalized]
    
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
    timeindex_2 = timeindex[test_y00 >=threshold_normalized]
    
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
    
    
    #
    y_test_rescale0 = re_sort(y_test_rescale1,y_test_rescale2,timeindex_1,timeindex_2)
    #y_test_rescale0 = np.hstack([y_test_rescale1,y_test_rescale2])
    pred_test_rescale0 = re_sort(pred_test_rescale1,pred_test_rescale2,timeindex_1,timeindex_2)
    #pred_test_rescale0 = np.hstack([pred_test_rescale1,pred_test_rescale2])
    
    res = [y_test_rescale0,pred_test_rescale0,
           y_test_rescale1,pred_test_rescale1,
           y_test_rescale2,pred_test_rescale2]
    return res

def get_prediction_past15():
    
    k = 1
    ################
    #2.prepressing##
    ################
    dataset_k1 = lstm_preprocessing(picklepath,wi)
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
    res_dict = {"model":None,
                "weekday":None,
                "test_mse_overall":None,
                "test_mse_freeflow":None,
                "test_mse_congestion":None,
                "test_mape_overall":None,
                "test_mape_freeflow":None,
                "test_mape_congestion":None}
    
    for ki,vi in zip(res_dict.keys(),l):
        res_dict[ki] = vi
    return res_dict

if __name__ == '__main__':
        thresholds = thresholds
        
        
        res1 = get_prediction_xgb()
        res2 = get_prediction_xgb2()
        os.exit()
        res3 = get_prediction_past15()
        #画free结果的
        os.exit()
        plt.close()
        #画Free的overall结果 看看延迟有没有缓解
        L = len(res2[0])
        
        plt.plot(list(range(len(res1[2]))),res1[3],label = 'xgb')
        plt.figure()
        plt.subplot(221)

        plt.plot(list(range(len(res1[2]))),res1[3],label = 'xgb')
        plt.plot(list(range(len(res1[2]))),res2[3],label = 'xgb2')
        plt.plot(list(range(len(res3[3]))),res3[3],label = 'past15')
        plt.plot(list(range(len(res1[2]))),res1[2],label = 'true')
        plt.legend()
        plt.title("Free Flow at weekday {}".format(wi+1))
        #画所有cong结果
        plt.subplot(222)

        plt.plot(list(range(len(res1[4]))),res1[5],label = 'xgb')
        plt.plot(list(range(len(res1[4]))),res2[5],label = 'xgb2')
        plt.plot(list(range(len(res3[5]))),res3[5],label = 'past15')
        plt.plot(list(range(len(res1[4]))),res1[4],label = 'true')
        plt.legend()
        plt.title("Congestion over threshold = {}".format(threshold))
        #得到mse
        mse1 = get_mse(res1)
        mse2 = get_mse(res2)
        mse3 = get_mse(res3)
        #画mse直方图
        plt.subplot(223)
        p = np.array([0,1,2])
        width_val = 0.25
        X,Y = p,mse1
        plt.bar(X,Y,label = 'xgb',alpha=0.6,width = width_val)
        for x,y in zip(X,Y):
            plt.text(x+0.05,y+0.05,'%.2f' %y, ha='center',va='bottom')
        X,Y = p+width_val,mse2
        plt.bar(X,Y,label = 'xgb2',alpha=0.6,width = width_val)
        for x,y in zip(X,Y):
            plt.text(x+0.05,y+0.05,'%.2f' %y, ha='center',va='bottom')
        plt.xticks(X,["overall","freeflow","congestion"])
        X,Y = p+width_val*2,mse3
        plt.bar(X,Y,label = 'past15',alpha=0.6,width = width_val)
        for x,y in zip(X,Y):
            plt.text(x+0.05,y+0.05,'%.2f' %y, ha='center',va='bottom')
        plt.legend()
        plt.title("MSE")

        #画mape直方图
        mape1 = get_mape(res1)
        mape2 = get_mape(res2)
        mape3 = get_mape(res3)
        
        plt.subplot(224)
        p = np.array([0,1,2])
        width_val = 0.25
        X,Y = p,mape1
        plt.bar(X,Y,label = 'xgb',alpha=0.6,width = width_val)
        for x,y in zip(X,Y):
            plt.text(x,y,'%.4f' %y, ha='center',va='bottom')
        X,Y = p+width_val,mape2
        plt.bar(X,Y,label = 'xgb2',alpha=0.6,width = width_val)
        for x,y in zip(X,Y):
            plt.text(x,y,'%.4f' %y, ha='center',va='bottom')
        plt.xticks(X,["overall","freeflow","congestion"])
        X,Y = p+width_val*2,mape3
        plt.bar(X,Y,label = 'past15',alpha=0.6,width = width_val)
        for x,y in zip(X,Y):
            plt.text(x,y,'%.4f' %y, ha='center',va='bottom')
        plt.legend()
        plt.title("MAPE")
        
        
        