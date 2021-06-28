# Descrpition of codes
### zjq_PreML.py
prepocessing data: divide the dataset into training and test set,select input and output features, dropout abnormal days, standardlization.  
### zjq_PreML_kfcv.py
precessing data: just like zjq_PreML.py but divide data set for k-fold cross validation

### zjq_XGB.py
use processed data to train a XGBoost model and get results of the congestion data and non-congestion data respectively.

### zjq_XGB_double_model.py
use processed data to train two XGBoost model on the the congestion data and non-congestion data and get results respectively.

### zjq_multimodel_kfcv.py
train single-XGBoost model, double-XGBoost model and also use baseline (past 15min median latency), to see the performance on the congestion data and non-congestion data with k-fold cross validation.

### zjq_LSTM.py
use processed data to train an LSTM model and get results 

