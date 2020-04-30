# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 07:24:41 2020

@author: rullyudin234@gmail.com
"""

# =============================================================================
# Melakukan peramalan 7 bulan ke depan terhadap data inflasi 
# menggunakan metode RBFNN (Radial Basis Function Neural Network) 
# =============================================================================

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from module_rbflayer import RBFLayer, InitCentersRandom
from module_kmeans_initializer import InitCentersKMeans
        
def periode(comodity_data, periode_numb):
    data_lag = pd.DataFrame()
    for i in range(output+periode_numb):
        data_lag['lag-'+str(i)] = comodity_data.iloc[:,0].shift(-i)
    return data_lag

def load_data(period):
    data = pd.read_excel("data/inflasi.xlsx")

    k=0
    for i in data.iloc[:,0]:
        month = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli',
                 'Agustus', 'September', 'Oktober', 'Nopember', 'Desember']
        if type(i) == str:
            split = i.split(" ")
            temp_month = [str(month.index(split[0])+1), split[1]]
            temp_month = '-'.join(x for x in temp_month)
            data.iloc[k:k+1, 0:1] = pd.to_datetime(temp_month, format='%m-%Y')
        k+=1
        
    data.set_index('Month', inplace=True)
    
    data = pd.DataFrame(data)
    data = periode(data, period)
    data.set_index(data.index, inplace=True)
    return data

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=1/5, 
                                                        random_state=0, 
                                                        shuffle=False)
    return X_train, X_test, y_train, y_test

def mape(y, y_pred):
    return np.mean(np.abs((y-y_pred)/y))
    
if __name__ == "__main__":

    iteration = 3
    
    y_test_denormalization_list = list()
    y_pred_denormalization_list = list()
    mape_list = list()
    rmse_list = list()
    time_list = list()
    
    output = 7 
    
    for j in range(1, output+1):
        period = j
        
        data_bm = load_data(period)
        data_bm_lag = data_bm.iloc[:-output-period+1,:]
        
        X = data_bm_lag.iloc[:, :period]
        y = data_bm_lag.iloc[:, period:]
    
        X_train, X_test, y_train, y_test = split(X, y)
        
        for node in range(1,iteration+1):
            print(node)
            # get time
            start = time.time()
            # modelling
            vars()['model'+str(j)] = Sequential()
            vars()['model'+str(j)].add(RBFLayer(node,
                                       InitCentersKMeans(X_train),
                                       input_shape=(period,)))
            vars()['model'+str(j)].add(Dense(output)) # depend on total dependent variable
            
            vars()['model'+str(j)].compile(loss='mean_squared_error',
                                          optimizer=RMSprop(),
                                          metrics=['accuracy', 'mse', 'mae'])
            
            history = vars()['model'+str(j)].fit(X_train, y_train,
                                                batch_size=50,
                                                epochs=200,
                                                verbose=1)
            y_pred = vars()['model'+str(j)].predict(X_test)
            # get time
            end = time.time()
            time_total = end-start 
            
            y_pred_denormalize = y_pred   
            y_test_denormalize = y_test
            
            #error_graph()
            
            # MAPE
            new_mape = mape(y_test_denormalize, y_pred_denormalize)
            # RMSE
            new_rmse = np.sqrt(mean_squared_error(
                    y_test_denormalize, y_pred_denormalize))
            
            y_test_denormalization_list.append(pd.DataFrame(y_test_denormalize).
                                               set_index(y_test.index))
            y_pred_denormalization_list.append(pd.DataFrame(y_pred_denormalize).
                                               set_index(y_test.index))
            mape_list.append(new_mape)
            rmse_list.append(new_rmse)
            time_list.append(time_total)
    
    # Get the best Period
    best_period = 1
    
    forecast_result = vars()['model'+str(best_period)].predict(
            data_bm.iloc[data_bm.shape[0]-best_period-1:data_bm.shape[0]-best_period,0:best_period])












