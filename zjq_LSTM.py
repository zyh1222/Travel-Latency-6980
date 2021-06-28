# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:57:46 2021

@author: ZJQ
"""
import os,sys
sys.path.append(os.getcwd())
from zjq_PreML import *
import time
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as torchdata
import datetime
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# 7:1:2
NUM_EPOCHES = 500
# BATCH_SIZE = 1020
HIDDEN_SIZE = 100
NUM_LAYER_LSTM = 2
#######################
##1.Define LSTM Model##
#######################
class lstm(nn.Module):
    def __init__(self,input_size=2,hidden_size=4,output_size=1,num_layer=2):
        super(lstm,self).__init__()
        
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer)
        self.layer2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x

def reshape_and_tensorlize(arr,dim):
    a1 = arr.reshape(-1, 1, dim)
    return torch.from_numpy(a1).float()

if __name__ == '__main__':
    picklepath = 'D:/DDMstudy/MSDM6980/Data/df_median_speed_starttime.pickle'
    #df = pd.read_pickle(picklepath)
    #for wi in range(3,4):
        
    wi = 1#0-4：Mon-Fri
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
    #for ki in range(k):
    ki = 0
    data = dataset[ki]
    dim = len(data[0][0])
    train_x = reshape_and_tensorlize(data[0],dim)
    train_y = reshape_and_tensorlize(data[1],1)
    test_x = reshape_and_tensorlize(data[2],dim)
    test_y = reshape_and_tensorlize(data[3],1)
    trainset = torchdata.TensorDataset(train_x,train_y)
    ########################
    #3. initialize model####
    ########################
    #batch number: 1
    BATCH_SIZE = len(train_x)
    loader = torchdata.DataLoader(
        dataset=trainset,
        batch_size=BATCH_SIZE,
        shuffle = False,
        #num_workers=3,
        num_workers=0,
    )
    #lstm(input_size=2,hidden_size=4,output_size=1,num_layer=2)
    model = lstm(dim,HIDDEN_SIZE,1,NUM_LAYER_LSTM)
    
    model.cuda()
    #optimizing target
    criterion = nn.MSELoss()
    #Gradient descent
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    #Setting gradually dropping learning rate
    scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=NUM_EPOCHES//5, gamma=0.8)
    start = time.time()
    # for e in range(NUM_EPOCHES):
    #     for step,(batch_x,batch_y) in enumerate(loader):
    #         print('epoch:{}|num_batch:{}|batch_size:{}'.format(e,step,len(batch_x)))
    for e in range(NUM_EPOCHES):
        for step,(batch_x,batch_y) in enumerate(loader):
            train_var_x = Variable(batch_x).cuda()#转变为torch.tensor
            train_var_y = Variable(batch_y).cuda()
            # 
            out = model(train_var_x)#训练模型
            loss = criterion(out, train_var_y)#计算损失
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(step)
        # if (e + 1) % 10 == 0:
        #     print(e+1)
        if (e + 1) % 5 == 0: # 每 5 次输出结果
            print('Epoch: {}, Loss: {:.5f},lr:{:.5f}'.format(e + 1, loss.data.item(),scheduler.get_last_lr()[0]))
    print('training time',time.time()-start)
    
    model = model.eval() # 转换成测试模式
    train_var_x = Variable(train_x).cuda()
    pred_train = model(train_var_x)
    pred_train = pred_train.cpu().view(-1).data.numpy()
    
    
    test_var_x = Variable(test_x).cuda()
    pred_test = model(test_var_x) # 测试集的预测结果
    pred_test = pred_test.cpu().view(-1).data.numpy()
    
    plt.figure()
    plt.plot(data[3].reshape(1,-1)[0],label='real')
    plt.plot(pred_test,label='pred')
    plt.legend()
    std = data[4][1].values[0]
    u = data[4][0].values[0]
    #rescale the normalized data
    y_train_rescale = data[1].reshape(1,-1)[0]*std+u
    pred_train_rescale = pred_train*std+u
    y_test_rescale = data[3].reshape(1,-1)[0]*std+u
    pred_test_rescale = pred_test*std+u
    l_mse.append((mean_squared_error(y_train_rescale,pred_train_rescale),mean_squared_error(y_test_rescale,pred_test_rescale)))
    l_mape.append((mean_absolute_percentage_error(y_train_rescale,pred_train_rescale),mean_absolute_percentage_error(y_test_rescale,pred_test_rescale)))
    
    
    print(l_mse[-1])
    print(l_mape[-1])
    
    #save prediction data
    test_data = np.vstack([pred_test_rescale,y_test_rescale])
    df_test_data = pd.DataFrame(test_data.T,columns = ['y_pred','y_true'])
    save_path = 'D:/DDMstudy/MSDM6980/Git/model_and_pred_data/'
    df_test_data.to_csv(save_path+'pred_3.14.csv')
    #save trained model
    torch.save(model.state_dict(), save_path+'lstm_params.pth')
    #model.load_state_dict(torch.load(save_path+'lstm_params.pth'))
    
