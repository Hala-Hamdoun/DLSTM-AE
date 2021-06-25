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



df=pd.read_csv('australia.csv')

daily_df=df.iloc[:,0:56]

diff3=daily_df.diff()
#diff3=diff3.values
 
n_past = 1
n_future = 1
n_features = 56

"""Train - Test Split"""

train_df,test_df = diff3[0:12], diff3[12:]  
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
rawdatay = rawdatay[12:, :]






model = Sequential()

#model.add(LSTM(15, input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))
model.add(LSTM(10, input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))
#model.add(LSTM(7, input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))
model.add(LSTM(7))
model.add(RepeatVector(n_future))
model.add(LSTM(7, return_sequences=True))
#model.add(LSTM(7, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(15, return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
model.summary()

""" **Training the models**"""





history_aus=model.fit(X_train,y_train,epochs=300,validation_split=0.33,batch_size=30,verbose=2,shuffle=False)

"""Saving model"""
model.save("aus(11).h5")


"""Predictions"""


pred_aus=model.predict(X_test)



"""Inverse Scaling of the predicted values"""

for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]  
    pred_aus[:,:,index]=scaler.inverse_transform(pred_aus[:,:,index])

    y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
    y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])





pred_aus = [inverse_difference( pred_aus[i],  rawdatay[i] ) for i in range(len(pred_aus))]
pred_aus=numpy.array(pred_aus)

print(pred_aus.shape)
print(rawdatay.shape)
print('MAPE', (MAPE(rawdatay, pred_aus))[0,:].mean())
'''
print('MAPE', (MAPE(rawdatay, pred_aus))[1,:].mean())

print('MAPE', (MAPE(rawdatay, pred_aus))[2,:].mean())

print('MAPE', (MAPE(rawdatay, pred_aus))[3,:].mean())

print('MAPE', (MAPE(rawdatay, pred_aus))[4,:].mean())

print('MAPE', (MAPE(rawdatay, pred_aus))[5,:].mean())

print('MAPE', (MAPE(rawdatay, pred_aus))[6,:].mean())

print('MAPE', (MAPE(rawdatay, pred_aus))[7,:].mean())

'''


#################################### """"" LEVEL TWO """"" ############
'''
daily_df=df.iloc[:,56:84]

diff3=daily_df.diff()
#diff3=diff3.values
 
n_past = 1
n_future = 1
n_features = 28

"""Train - Test Split"""

train_df,test_df = diff3[0:12], diff3[12:]  


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

z_train=np.zeros((12-n_future,n_past,28))
X_train = np.concatenate([X_train, z_train], -1)
print(X_train.shape)
print(y_train.shape)
z_test=np.zeros((24-n_future,n_past,28))
X_test = np.concatenate([X_test, z_test], -1)
print(X_test.shape)
print(y_test.shape)



rawdataX,rawdatay=split_series(daily_df.values,n_past, n_future)
rawdatay = rawdatay[12:, :]




"""Load the saved model"""

from keras.models import Model
from keras.models import load_model
model2=load_model('aus(1steps).h5')

for layer in model2.layers[:5]:
    layer.trainable = False

# Get input
new_input = model2.input
# Find the layer to connect
hidden_layer = model2.layers[-1].output
# Connect a new layer on it
new_output = Dense(n_features) (hidden_layer)
# Build a new model
new_model2 = Model(new_input, new_output)
new_model2.compile(optimizer='adam', loss='mse')
new_model2.summary()


fit_history2 = new_model2.fit(X_train, y_train, epochs=250, batch_size=50, 
                     validation_split=0.33, verbose=2, shuffle=False) 




"""Predictions"""


pred_aus=new_model2.predict(X_test)
print(pred_aus.shape)



"""Inverse Scaling of the predicted values"""

for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]  
    pred_aus[:,:,index]=scaler.inverse_transform(pred_aus[:,:,index])

   # y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
   # y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])

red_aus = [inverse_difference( pred_aus[i],  rawdatay[i] ) for i in range(len(pred_aus))]
pred_aus=numpy.array(pred_aus)
print(pred_aus)
print(pred_aus.shape)



"""Error Estimation"""

print('MAPE', (MAPE(rawdatay, pred_aus))[0,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[0,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[0,:].mean())

print('MAPE', (MAPE(rawdatay, pred_aus))[1,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[1,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[1,:].mean())


print('MAPE', (MAPE(rawdatay, pred_aus))[2,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[2,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[2,:].mean())


print('MAPE', (MAPE(rawdatay, pred_aus))[3,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[3,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[3,:].mean())


print('MAPE', (MAPE(rawdatay, pred_aus))[4,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[4,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[4,:].mean())

print('MAPE', (MAPE(rawdatay, pred_aus))[5,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[5,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[5,:].mean())


print('MAPE', (MAPE(rawdatay, pred_aus))[6,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[6,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[6,:].mean())


print('MAPE', (MAPE(rawdatay, pred_aus))[7,:].mean())

print('RMSE', (RMSE(rawdatay, pred_aus))[7,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[7,:].mean())
#################################### """"" LEVEL One """"" ############
'''
'''
daily_df=df.iloc[:,84:88]

diff3=daily_df.diff()
#diff3=diff3.values
 
n_past = 1
n_future = 8
n_features =4

"""Train - Test Split"""

train_df,test_df = diff3[0:12], diff3[12:] 

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

z_train=np.zeros((12-n_future,n_past,52))
X_train = np.concatenate([X_train, z_train], -1)
print(X_train.shape)
print(y_train.shape)
z_test=np.zeros((24-n_future,n_past,52))
X_test = np.concatenate([X_test, z_test], -1)
print(X_test.shape)
print(y_test.shape)



rawdataX,rawdatay=split_series(daily_df.values,n_past, n_future)
rawdatay = rawdatay[12:, :]




from keras.models import Model
from keras.models import load_model
model2=load_model('aus(8steps).h5') #######################

for layer in model2.layers[:5]:
    layer.trainable = False

# Get input
new_input = model2.input
# Find the layer to connect
hidden_layer = model2.layers[-2].output
# Connect a new layer on it
new_output = Dense(n_features) (hidden_layer)
# Build a new model
new_model2 = Model(new_input, new_output)
new_model2.compile(optimizer='adam', loss='mse')
new_model2.summary()


fit_history2 = new_model2.fit(X_train, y_train, epochs=500, batch_size=60, 
                     validation_split=0.33, verbose=2, shuffle=False) 




"""Predictions"""


pred_aus=new_model2.predict(X_test)
print(pred_aus.shape)



"""Inverse Scaling of the predicted values"""

for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]  
    pred_aus[:,:,index]=scaler.inverse_transform(pred_aus[:,:,index])

   # y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
   # y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])







pred_aus = [inverse_difference( pred_aus[i],  rawdatay[i] ) for i in range(len(pred_aus))]
pred_aus=numpy.array(pred_aus)
print(pred_aus)
print(pred_aus.shape)




"""Error Estimation"""
print('MAPE', (MAPE(rawdatay, pred_aus))[0,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[0,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[0,:].mean())



print('MAPE', (MAPE(rawdatay, pred_aus))[1,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[1,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[1,:].mean())


print('MAPE', (MAPE(rawdatay, pred_aus))[2,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[2,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[2,:].mean())


print('MAPE', (MAPE(rawdatay, pred_aus))[3,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[3,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[3,:].mean())


print('MAPE', (MAPE(rawdatay, pred_aus))[4,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[4,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[4,:].mean())


print('MAPE', (MAPE(rawdatay, pred_aus))[5,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[5,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[5,:].mean())


print('MAPE', (MAPE(rawdatay, pred_aus))[6,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[6,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[6,:].mean())

print('MAPE', (MAPE(rawdatay, pred_aus))[7,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[7,:].mean())
print('dRMSE', (dRMSE(rawdatay, pred_aus))[7,:].mean())

'''
#################################### """"" LEVEL One """"" ############
'''
daily_df=df.iloc[:,88:89]

diff3=daily_df.diff()
#diff3=diff3.values
 
n_past = 1
n_future = 8
n_features =1

"""Train - Test Split"""

train_df,test_df = diff3[0:12], diff3[12:] 


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

z_train=np.zeros((12-n_future,n_past,55))
X_train = np.concatenate([X_train, z_train], -1)
print(X_train.shape)
print(y_train.shape)
z_test=np.zeros((24-n_future,n_past,55))
X_test = np.concatenate([X_test, z_test], -1)
print(X_test.shape)
print(y_test.shape)


print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk', X_train.shape)
rawdataX,rawdatay=split_series(daily_df.values,n_past, n_future)
rawdatay = rawdatay[12:, :]




from keras.models import Model
from keras.models import load_model
model2=load_model('aus(8steps).h5') #######################

for layer in model2.layers[:7]:
    layer.trainable = False

# Get input
new_input = model2.input
# Find the layer to connect
hidden_layer = model2.layers[-2].output
# Connect a new layer on it
new_output = Dense(n_features) (hidden_layer)
# Build a new model
new_model2 = Model(new_input, new_output)
new_model2.compile(optimizer='adam', loss='mse')
new_model2.summary()


fit_history2 = new_model2.fit(X_train, y_train, epochs=450, batch_size=60, 
                     validation_split=0.33, verbose=2, shuffle=False)   #700




"""Predictions"""


pred_aus=new_model2.predict(X_test)
print(pred_aus.shape)



"""Inverse Scaling of the predicted values"""

for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]  
    pred_aus[:,:,index]=scaler.inverse_transform(pred_aus[:,:,index])

   # y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
   # y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])


pred_aus = [inverse_difference( pred_aus[i],  rawdatay[i] ) for i in range(len(pred_aus))]
pred_aus=numpy.array(pred_aus)
print(pred_aus)
print(pred_aus.shape)



def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

dy_true = difference(rawdatay)
dy_pred = difference(pred_aus)



"""Error Estimation"""
print('MAPE', (MAPE(rawdatay, pred_aus))[0,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[0,:].mean())
print('dRMSE', (dRMSE(dy_true, dy_pred))[0,:])

print('MAPE', (MAPE(rawdatay, pred_aus))[1,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[1,:].mean())
print('dRMSE', (dRMSE(dy_true, dy_pred))[1,:])


print('MAPE', (MAPE(rawdatay, pred_aus))[2,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[2,:].mean())
print('dRMSE', (dRMSE(dy_true, dy_pred))[2,:])


print('MAPE', (MAPE(rawdatay, pred_aus))[3,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[3,:].mean())
print('dRMSE', (dRMSE(dy_true, dy_pred))[3,:])


print('MAPE', (MAPE(rawdatay, pred_aus))[4,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[4,:].mean())
print('dRMSE', (dRMSE(dy_true, dy_pred))[4,:])


print('MAPE', (MAPE(rawdatay, pred_aus))[5,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[5,:].mean())
print('dRMSE', (dRMSE(dy_true, dy_pred))[5,:])


print('MAPE', (MAPE(rawdatay, pred_aus))[6,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[6,:].mean())
print('dRMSE', (dRMSE(dy_true, dy_pred))[6,:])

print('MAPE', (MAPE(rawdatay, pred_aus))[7,:].mean())
print('RMSE', (RMSE(rawdatay, pred_aus))[7,:].mean())
print('dRMSE', (dRMSE(dy_true, dy_pred))[7,:])
'''
