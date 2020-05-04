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
        data_lag['lag-'+str(i)] = comodity_data.iloc[:,0].shift(i)
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

def best_mape():
    period_mape_list = list()
    for i in range(0,len(mape_list)-iteration+1,iteration):
        period_mape_list.append(min(mape_list[i:i+iteration])*100)
    return period_mape_list

def error_graph():
    plt.plot(history.history['mse'], label='mse')
    plt.plot(history.history['mae'], label='mae')
    plt.legend(loc='upper right')
    plt.show()
    
if __name__ == "__main__":

    y_test_denormalization_list = list()
    y_pred_denormalization_list = list()
    mape_list = list()
    rmse_list = list()
    time_list = list()
    
    iteration = 30
    output = 7
    period = 1 #input manual
    
    data_bm = load_data(period)
    data_bm = data_bm.iloc[::-1]
    data_bm_lag = data_bm.iloc[:-output-period+1,:]
    
    X = data_bm_lag.iloc[:, :period]
    y = data_bm_lag.iloc[:, period:]

    X_train, X_test, y_train, y_test = split(X, y)
    
    for node in range(1, iteration+1):
        print(node)
        # get time
        start = time.time()
        # modelling
        vars()['model'+str(period)] = Sequential()
        vars()['model'+str(period)].add(RBFLayer(node,
                                   InitCentersKMeans(X_train),
                                   input_shape=(period,)))
        vars()['model'+str(period)].add(Dense(output)) # depend on total dependent variable
        
        vars()['model'+str(period)].compile(loss='mean_squared_error',
                                      optimizer=RMSprop(),
                                      metrics=['accuracy', 'mse', 'mae'])
        
        history = vars()['model'+str(period)].fit(X_train, y_train,
                                            batch_size=50,
                                            epochs=200,
                                            verbose=1)
        y_pred = vars()['model'+str(period)].predict(X_test)
        # get time
        end = time.time()
        time_total = end-start 
        
        y_pred_denormalization = np.array(y_pred)
        y_test_denormalization = np.array(y_test)
        
        #error_graph()
        
        # MAPE
        new_mape = mape(y_test_denormalization, y_pred_denormalization)
        # RMSE
        new_rmse = np.sqrt(mean_squared_error(
                y_test_denormalization, y_pred_denormalization))
        
        y_test_denormalization_list.append(pd.DataFrame(y_test_denormalization).
                                           set_index(y_test.index))
        y_pred_denormalization_list.append(pd.DataFrame(y_pred_denormalization).
                                           set_index(y_test.index))
        mape_list.append(new_mape)
        rmse_list.append(new_rmse)
        time_list.append(time_total)

    # Period DataFrame
    period_df = pd.DataFrame({'Period':[i for i in range(1,period+1)],
                                        'MAPE':best_mape()}).set_index('Period')
    # Period Graph
    plt.plot(period_df)
    plt.ylabel(period_df.columns[0])
    plt.xlabel(period_df.index.names[0])
    plt.show()
    
# Get the best Period
best_period = 1 #input manual

# Get minimal mape, rmse, and time in the 4th period
best_period_mape = min(mape_list[(best_period-1)*iteration:best_period*iteration])
best_period_rmse = min(rmse_list[(best_period-1)*iteration:best_period*iteration])
best_period_time = min(time_list[(best_period-1)*iteration:best_period*iteration])
print('best period mape: {}\nbest period rmse: {}\nbest period time: {}'.format(
        best_period_mape, best_period_rmse, best_period_time))

# Graph
period_graph(5)

forecast_result = vars()['model'+str(best_period)].predict(data_bm.iloc[
        data_bm.shape[0]-best_period:data_bm.shape[0]-best_period+1,0:best_period])

# Tidak cocok menggunakan metode RBFNN , karena mape terlalu besar.









