# -*- coding: utf-8 -*-
"""
Created on Sat May 5  2021

@author: Hala Hamdoun
"""

import pandas as pd
import numpy as np
from numpy import *
import tensorflow as tf
import os
import random as rn
import sys 
from numpy.random import seed
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas import Series
from pandas import concat
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from math import sqrt
import numpy 
from numpy import concatenate

#from keras.callbacks import ModelCheckpoint



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED'] = '0'
numpy.random.seed(42)
rn.seed(12345)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)


from tensorflow.python.keras import backend as k
#tf.set_random_seed(1234)
tf.random.set_seed(1234)

sess =  tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


def inverse_difference(last_ob, value):
	return value + last_ob


"""**Checking Error** """

from sklearn.metrics import mean_absolute_error
def MAPE(x,y):
	result=0
	for i in range(len(x)):
		result += abs((x[i]-y[i])/x[i])
	result /= len(x)
	result *= 100
	return result



def RMSE(x,y):
	result=0
	for i in range(len(x)):
		result += ((x[i]-y[i])/x[i])**2
	result /= len(x)
	#result = sqrt(result)

	return result



def dRMSE(y_true, y_pred):
	dy_true=np.diff(y_true, axis=0)
	dy_pred=np.diff(y_pred, axis=0)
	result=0
	for i in range(len(dy_true)):
		result += ((dy_true[i]-dy_pred[i])/dy_true[i])**2
	result /= len(dy_true)
	result = sqrt(result)
	return result


df=pd.read_csv('brazil.csv')

daily_df=df.iloc[:,0:14]

diff3=daily_df.diff()
#diff3=diff3.values
 
n_past = 1
n_future = 9
n_features = 14

"""Train - Test Split"""

train_df,test_df = diff3[0:13103], diff3[13103:]  # 75% and 25%
train_df.shape,test_df.shape

"""Scaling the values for faster training of the models."""

train = train_df
scalers={}

for i in train_df.columns:

    scaler = MinMaxScaler(feature_range=(0,1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    train[i]=s_s

test = test_df
for i in train_df.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+i] = scaler
    test[i]=s_s

"""**Converting the series to samples for supervised learning**"""

def split_series(series, n_past, n_future):
  #
  # n_past ==> no of past observations
  #
  # n_future ==> no of future observations 
  #
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    # slicing the past and future parts of the window
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)


X_train, y_train = split_series(train.values,n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
where_are_NaNs = isnan(X_train)
X_train[where_are_NaNs] = 0
X_test, y_test = split_series(test.values,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))


rawdataX,rawdatay=split_series(daily_df.values,n_past, n_future)
rawdatay = rawdatay[13103:, :]

#Model#
model = Sequential()
model.add(LSTM(10, input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))
model.add(LSTM(7, input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))
model.add(LSTM(4))
model.add(RepeatVector(n_future))
model.add(LSTM(4, return_sequences=True))
model.add(LSTM(7, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
#model.summary()

""" **Training the models**"""
history_brazil=model.fit(X_train,y_train,epochs=40,validation_split=0.33,batch_size=400,verbose=2,shuffle=False)


"""Saving model"""
model.save("brazil(9s).h5")



"""Predictions"""
pred_brazil=model.predict(X_test)



"""Inverse Scaling of the predicted values"""

for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]  
    pred_brazil[:,:,index]=scaler.inverse_transform(pred_brazil[:,:,index])

    y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
    y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])

pred_brazil = [inverse_difference( pred_brazil[i],  rawdatay[i] ) for i in range(len(pred_brazil))]
pred_brazil=numpy.array(pred_brazil)


"""Error Estimation"""
'''
print(pred_brazil.shape)
print(rawdatay.shape)
print('MAPE', (MAPE(rawdatay, pred_brazil))[0,:].mean())

print('MAPE', (MAPE(rawdatay, pred_brazil))[1,:].mean())
print('2step', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean())/2)

print('MAPE', (MAPE(rawdatay, pred_brazil))[2,:].mean())
print('3step', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean())/3)

print('MAPE', (MAPE(rawdatay, pred_brazil))[3,:].mean())
print('4step', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean())/4)

print('MAPE', (MAPE(rawdatay, pred_brazil))[4,:].mean())
print('5step', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean())/5)


print('MAPE', (MAPE(rawdatay, pred_brazil))[5,:].mean())
print('6step', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean()+(MAPE(rawdatay, pred_brazil))[5,:].mean())/6)

print('MAPE', (MAPE(rawdatay, pred_brazil))[6,:].mean())
print('7step', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean()+(MAPE(rawdatay, pred_brazil))[5,:].mean()+(MAPE(rawdatay, pred_brazil))[6,:].mean())/7)

print('MAPE', (MAPE(rawdatay, pred_brazil))[7,:].mean())
print('8step', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean()+(MAPE(rawdatay, pred_brazil))[5,:].mean()+(MAPE(rawdatay, pred_brazil))[6,:].mean()+(MAPE(rawdatay, pred_brazil))[7,:].mean())/8)

print('MAPE', (MAPE(rawdatay, pred_brazil))[8,:].mean())
print('9step', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean()+(MAPE(rawdatay, pred_brazil))[5,:].mean()+(MAPE(rawdatay, pred_brazil))[6,:].mean()+(MAPE(rawdatay, pred_brazil))[7,:].mean()+(MAPE(rawdatay, pred_brazil))[8,:].mean())/9)
'''
#################################### """"" LEVEL TWO """"" ############

'''
daily_df=df.iloc[:,14:18]

diff3=daily_df.diff()
#diff3=diff3.values
 
n_past = 1
n_future =9
n_features = 4

"""Train - Test Split"""

train_df,test_df = diff3[0:13103], diff3[13103:]  # 75% and 25%


"""Scaling the values for faster training of the models."""

train = train_df
scalers={}

for i in train_df.columns:

    scaler = MinMaxScaler(feature_range=(0,1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    train[i]=s_s

test = test_df
for i in train_df.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+i] = scaler
    test[i]=s_s


"""**Converting the series to samples for supervised learning**"""

def split_series(series, n_past, n_future):
  #
  # n_past ==> no of past observations
  #
  # n_future ==> no of future observations 
  #
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    # slicing the past and future parts of the window
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)


X_train, y_train = split_series(train.values,n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
where_are_NaNs = isnan(X_train)
X_train[where_are_NaNs] = 0

X_test, y_test = split_series(test.values,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
print(y_train,X_train)


"""Zero Padding"""

z_train=np.zeros((13103-n_future,n_past,10))
X_train = np.concatenate([X_train, z_train], -1)
print(X_train.shape)
print(y_train.shape)
z_test=np.zeros((4417-n_future,n_past,10))
X_test = np.concatenate([X_test, z_test], -1)
print(X_test.shape)
print(y_test.shape)



rawdataX,rawdatay=split_series(daily_df.values,n_past, n_future)
rawdatay = rawdatay[13103:, :]


"""Load the saved model"""

from keras.models import Model
from keras.models import load_model
model2=load_model('brazil(9s).h5')

for layer in model2.layers[:7]:
    layer.trainable = False

# Get input
new_input = model2.input
# Find the layer to connect
hidden_layer = model2.layers[-2].output
# Connect a new layer on it
new_output = Dense(4) (hidden_layer)
# Build a new model
new_model2 = Model(new_input, new_output)
new_model2.compile(optimizer='adam', loss='mse')
new_model2.summary()


fit_history2 = new_model2.fit(X_train, y_train, epochs=30, batch_size=100, 
                     validation_split=0.33, verbose=2, shuffle=False) 




"""Predictions"""


pred_brazil=new_model2.predict(X_test)
print(pred_brazil.shape)



"""Inverse Scaling of the predicted values"""

for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]  
    pred_brazil[:,:,index]=scaler.inverse_transform(pred_brazil[:,:,index])

   # y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
   # y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])

pred_brazil = [inverse_difference( pred_brazil[i],  rawdatay[i] ) for i in range(len(pred_brazil))]
pred_brazil=numpy.array(pred_brazil)
print(pred_brazil)


"""Error Estimation"""

print('MAPE', (MAPE(rawdatay, pred_brazil))[0,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[0,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[0,:].mean())

print('MAPE', (MAPE(rawdatay, pred_brazil))[1,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[1,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[1,:].mean())

print('2step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean())/2)
print('2step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean())/2)
print('2step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean())/2)


print('MAPE', (MAPE(rawdatay, pred_brazil))[2,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[2,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[2,:].mean())
print('3step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean())/3)
print('3step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean())/3)
print('3step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean())/3)

print('MAPE', (MAPE(rawdatay, pred_brazil))[3,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[3,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[3,:].mean())

print('4step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean())/4)
print('4step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean()+(RMSE(rawdatay, pred_brazil))[3,:].mean())/4)
print('4step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean()+(dRMSE(rawdatay, pred_brazil))[3,:].mean())/4)


print('MAPE', (MAPE(rawdatay, pred_brazil))[4,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[4,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[4,:].mean())



print('5step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean())/5)
print('5step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean()+(RMSE(rawdatay, pred_brazil))[3,:].mean()+(RMSE(rawdatay, pred_brazil))[4,:].mean())/5)
print('5step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean()+(dRMSE(rawdatay, pred_brazil))[3,:].mean()+(dRMSE(rawdatay, pred_brazil))[4,:].mean())/5)



print('MAPE', (MAPE(rawdatay, pred_brazil))[5,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[5,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[5,:].mean())

print('6step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean()+(MAPE(rawdatay, pred_brazil))[5,:].mean())/6)
print('6step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean()+(RMSE(rawdatay, pred_brazil))[3,:].mean()+(RMSE(rawdatay, pred_brazil))[4,:].mean()+(RMSE(rawdatay, pred_brazil))[5,:].mean())/6)
print('6step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean()+(dRMSE(rawdatay, pred_brazil))[3,:].mean()+(dRMSE(rawdatay, pred_brazil))[4,:].mean()+(dRMSE(rawdatay, pred_brazil))[5,:].mean())/6)


print('MAPE', (MAPE(rawdatay, pred_brazil))[6,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[6,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[6,:].mean())
print('7step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean()+(MAPE(rawdatay, pred_brazil))[5,:].mean()+(MAPE(rawdatay, pred_brazil))[6,:].mean())/7)
print('7step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean()+(RMSE(rawdatay, pred_brazil))[3,:].mean()+(RMSE(rawdatay, pred_brazil))[4,:].mean()+(RMSE(rawdatay, pred_brazil))[5,:].mean()+(RMSE(rawdatay, pred_brazil))[6,:].mean())/7)
print('7step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean()+(dRMSE(rawdatay, pred_brazil))[3,:].mean()+(dRMSE(rawdatay, pred_brazil))[4,:].mean()+(dRMSE(rawdatay, pred_brazil))[5,:].mean()+(dRMSE(rawdatay, pred_brazil))[6,:].mean())/7)

print('MAPE', (MAPE(rawdatay, pred_brazil))[7,:].mean())

print('RMSE', (RMSE(rawdatay, pred_brazil))[7,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[7,:].mean())

print('8step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean()+(MAPE(rawdatay, pred_brazil))[5,:].mean()+(MAPE(rawdatay, pred_brazil))[6,:].mean()+(MAPE(rawdatay, pred_brazil))[7,:].mean())/8)
print('8step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean()+(RMSE(rawdatay, pred_brazil))[3,:].mean()+(RMSE(rawdatay, pred_brazil))[4,:].mean()+(RMSE(rawdatay, pred_brazil))[5,:].mean()+(RMSE(rawdatay, pred_brazil))[6,:].mean()+(RMSE(rawdatay, pred_brazil))[7,:].mean())/8)
print('8step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean()+(dRMSE(rawdatay, pred_brazil))[3,:].mean()+(dRMSE(rawdatay, pred_brazil))[4,:].mean()+(dRMSE(rawdatay, pred_brazil))[5,:].mean()+(dRMSE(rawdatay, pred_brazil))[6,:].mean()+(dRMSE(rawdatay, pred_brazil))[7,:].mean())/8)



print('MAPE', (MAPE(rawdatay, pred_brazil))[8,:].mean())

print('RMSE', (RMSE(rawdatay, pred_brazil))[8,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[8,:].mean())

print('8step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean()+(MAPE(rawdatay, pred_brazil))[5,:].mean()+(MAPE(rawdatay, pred_brazil))[6,:].mean()+(MAPE(rawdatay, pred_brazil))[7,:].mean()+(MAPE(rawdatay, pred_brazil))[8,:].mean())/9)
print('8step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean()+(RMSE(rawdatay, pred_brazil))[3,:].mean()+(RMSE(rawdatay, pred_brazil))[4,:].mean()+(RMSE(rawdatay, pred_brazil))[5,:].mean()+(RMSE(rawdatay, pred_brazil))[6,:].mean()+(RMSE(rawdatay, pred_brazil))[7,:].mean()+(RMSE(rawdatay, pred_brazil))[8,:].mean())/9)
print('8step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean()+(dRMSE(rawdatay, pred_brazil))[3,:].mean()+(dRMSE(rawdatay, pred_brazil))[4,:].mean()+(dRMSE(rawdatay, pred_brazil))[5,:].mean()+(dRMSE(rawdatay, pred_brazil))[6,:].mean()+(dRMSE(rawdatay, pred_brazil))[7,:].mean()+(dRMSE(rawdatay, pred_brazil))[8,:].mean())/9)
'''
#################################### """"" LEVEL ROOT """"" ############
'''
daily_df=df.iloc[:,18:19]

diff3=daily_df.diff()
#diff3=diff3.values
 
n_past = 1
n_future = 9

n_features = 1

"""Train - Test Split"""

train_df,test_df = diff3[0:13103], diff3[13103:]  # 75% and 25%


"""Scaling the values for faster training of the models."""

train = train_df
scalers={}

for i in train_df.columns:

    scaler = MinMaxScaler(feature_range=(0,1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    train[i]=s_s

test = test_df
for i in train_df.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+i] = scaler
    test[i]=s_s


"""**Converting the series to samples for supervised learning**"""

def split_series(series, n_past, n_future):
  #
  # n_past ==> no of past observations
  #
  # n_future ==> no of future observations 
  #
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    # slicing the past and future parts of the window
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)


X_train, y_train = split_series(train.values,n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
where_are_NaNs = isnan(X_train)
X_train[where_are_NaNs] = 0

X_test, y_test = split_series(test.values,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
print(y_train,X_train)


"""Zero Padding"""

z_train=np.zeros((13103-n_future,n_past,13))
X_train = np.concatenate([X_train, z_train], -1)
print(X_train.shape)
print(y_train.shape)
z_test=np.zeros((4417-n_future,n_past,13))
X_test = np.concatenate([X_test, z_test], -1)
print(X_test.shape)
print(y_test.shape)



rawdataX,rawdatay=split_series(daily_df.values,n_past, n_future)
rawdatay = rawdatay[13103:, :]




from keras.models import Model
from keras.models import load_model
model2=load_model('brazil(9s).h5')

for layer in model2.layers[:7]:
    layer.trainable = False

# Get input
new_input = model2.input
# Find the layer to connect
hidden_layer = model2.layers[-2].output
# Connect a new layer on it
new_output = Dense(1) (hidden_layer)
# Build a new model
new_model2 = Model(new_input, new_output)
new_model2.compile(optimizer='adam', loss='mse')
new_model2.summary()


fit_history2 = new_model2.fit(X_train, y_train, epochs=20, batch_size=100, 
                     validation_split=0.33, verbose=2, shuffle=False) 




"""Predictions"""


pred_brazil=new_model2.predict(X_test)
print(pred_brazil.shape)



"""Inverse Scaling of the predicted values"""

for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]  
    pred_brazil[:,:,index]=scaler.inverse_transform(pred_brazil[:,:,index])

   # y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
   # y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])




pred_brazil = [inverse_difference( pred_brazil[i],  rawdatay[i] ) for i in range(len(pred_brazil))]
pred_brazil=numpy.array(pred_brazil)
print(pred_brazil)
print(pred_brazil.shape)


"""Error Estimation"""

print('MAPE', (MAPE(rawdatay, pred_brazil))[0,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[0,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[0,:].mean())

print('MAPE', (MAPE(rawdatay, pred_brazil))[1,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[1,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[1,:].mean())

print('2step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean())/2)
print('2step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean())/2)
print('2step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean())/2)


print('MAPE', (MAPE(rawdatay, pred_brazil))[2,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[2,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[2,:].mean())
print('3step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean())/3)
print('3step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean())/3)
print('3step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean())/3)

print('MAPE', (MAPE(rawdatay, pred_brazil))[3,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[3,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[3,:].mean())

print('4step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean())/4)
print('4step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean()+(RMSE(rawdatay, pred_brazil))[3,:].mean())/4)
print('4step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean()+(dRMSE(rawdatay, pred_brazil))[3,:].mean())/4)


print('MAPE', (MAPE(rawdatay, pred_brazil))[4,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[4,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[4,:].mean())



print('5step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean())/5)
print('5step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean()+(RMSE(rawdatay, pred_brazil))[3,:].mean()+(RMSE(rawdatay, pred_brazil))[4,:].mean())/5)
print('5step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean()+(dRMSE(rawdatay, pred_brazil))[3,:].mean()+(dRMSE(rawdatay, pred_brazil))[4,:].mean())/5)



print('MAPE', (MAPE(rawdatay, pred_brazil))[5,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[5,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[5,:].mean())

print('6step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean()+(MAPE(rawdatay, pred_brazil))[5,:].mean())/6)
print('6step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean()+(RMSE(rawdatay, pred_brazil))[3,:].mean()+(RMSE(rawdatay, pred_brazil))[4,:].mean()+(RMSE(rawdatay, pred_brazil))[5,:].mean())/6)
print('6step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean()+(dRMSE(rawdatay, pred_brazil))[3,:].mean()+(dRMSE(rawdatay, pred_brazil))[4,:].mean()+(dRMSE(rawdatay, pred_brazil))[5,:].mean())/6)


print('MAPE', (MAPE(rawdatay, pred_brazil))[6,:].mean())
print('RMSE', (RMSE(rawdatay, pred_brazil))[6,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[6,:].mean())
print('7step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean()+(MAPE(rawdatay, pred_brazil))[5,:].mean()+(MAPE(rawdatay, pred_brazil))[6,:].mean())/7)
print('7step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean()+(RMSE(rawdatay, pred_brazil))[3,:].mean()+(RMSE(rawdatay, pred_brazil))[4,:].mean()+(RMSE(rawdatay, pred_brazil))[5,:].mean()+(RMSE(rawdatay, pred_brazil))[6,:].mean())/7)
print('7step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean()+(dRMSE(rawdatay, pred_brazil))[3,:].mean()+(dRMSE(rawdatay, pred_brazil))[4,:].mean()+(dRMSE(rawdatay, pred_brazil))[5,:].mean()+(dRMSE(rawdatay, pred_brazil))[6,:].mean())/7)

print('MAPE', (MAPE(rawdatay, pred_brazil))[7,:].mean())

print('RMSE', (RMSE(rawdatay, pred_brazil))[7,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[7,:].mean())

print('8step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean()+(MAPE(rawdatay, pred_brazil))[5,:].mean()+(MAPE(rawdatay, pred_brazil))[6,:].mean()+(MAPE(rawdatay, pred_brazil))[7,:].mean())/8)
print('8step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean()+(RMSE(rawdatay, pred_brazil))[3,:].mean()+(RMSE(rawdatay, pred_brazil))[4,:].mean()+(RMSE(rawdatay, pred_brazil))[5,:].mean()+(RMSE(rawdatay, pred_brazil))[6,:].mean()+(RMSE(rawdatay, pred_brazil))[7,:].mean())/8)
print('8step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean()+(dRMSE(rawdatay, pred_brazil))[3,:].mean()+(dRMSE(rawdatay, pred_brazil))[4,:].mean()+(dRMSE(rawdatay, pred_brazil))[5,:].mean()+(dRMSE(rawdatay, pred_brazil))[6,:].mean()+(dRMSE(rawdatay, pred_brazil))[7,:].mean())/8)



print('MAPE', (MAPE(rawdatay, pred_brazil))[8,:].mean())

print('RMSE', (RMSE(rawdatay, pred_brazil))[8,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_brazil))[8,:].mean())

print('8step-lvl2', ((MAPE(rawdatay, pred_brazil))[0,:].mean()+(MAPE(rawdatay, pred_brazil))[1,:].mean()+(MAPE(rawdatay, pred_brazil))[2,:].mean()+(MAPE(rawdatay, pred_brazil))[3,:].mean()+(MAPE(rawdatay, pred_brazil))[4,:].mean()+(MAPE(rawdatay, pred_brazil))[5,:].mean()+(MAPE(rawdatay, pred_brazil))[6,:].mean()+(MAPE(rawdatay, pred_brazil))[7,:].mean()+(MAPE(rawdatay, pred_brazil))[8,:].mean())/9)
print('8step-lvl2', ((RMSE(rawdatay, pred_brazil))[0,:].mean()+(RMSE(rawdatay, pred_brazil))[1,:].mean()+(RMSE(rawdatay, pred_brazil))[2,:].mean()+(RMSE(rawdatay, pred_brazil))[3,:].mean()+(RMSE(rawdatay, pred_brazil))[4,:].mean()+(RMSE(rawdatay, pred_brazil))[5,:].mean()+(RMSE(rawdatay, pred_brazil))[6,:].mean()+(RMSE(rawdatay, pred_brazil))[7,:].mean()+(RMSE(rawdatay, pred_brazil))[8,:].mean())/9)
print('8step-lvl2', ((dRMSE(rawdatay, pred_brazil))[0,:].mean()+(dRMSE(rawdatay, pred_brazil))[1,:].mean()+(dRMSE(rawdatay, pred_brazil))[2,:].mean()+(dRMSE(rawdatay, pred_brazil))[3,:].mean()+(dRMSE(rawdatay, pred_brazil))[4,:].mean()+(dRMSE(rawdatay, pred_brazil))[5,:].mean()+(dRMSE(rawdatay, pred_brazil))[6,:].mean()+(dRMSE(rawdatay, pred_brazil))[7,:].mean()+(dRMSE(rawdatay, pred_brazil))[8,:].mean())/9)
'''
