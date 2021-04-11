# -*- coding: utf-8 -*-
"""
@author: CHAGUER Badreddine
"""
import time
from tempfile import TemporaryFile
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from algorithms.data_utils import randomize_data,split_data, load_data
from algorithms.linear_regression import LinearRegression,LinearRegressionMajority 
from algorithms.linear_regression import LinearRegressionMedian,LinearRegressionMean
from algorithms.linear_regression import LinearRegressionLeastSquares
from algorithms.linear_regression import LinearRegressionRidge

######## Performance of a constant estimator ###############################

# Data tagged from the file YearPredictionMSD_100.npz
X_labeled, y_labeled, X_unlabeled= load_data('io\YearPredictionMSD_100.npz')

# Split train set and validation set
X_train, y_train, X_test, y_test = split_data(X_labeled, y_labeled,2/3)
y_train = y_train.reshape(np.shape(y_train)[0], 1)
y_test = y_test.reshape(np.shape(y_test)[0], 1)
S_train = np.concatenate((X_train, y_train), axis = 1)
S_valid = np.concatenate((X_test, y_test) , axis=1)

# N = {2**5, 2**6, ..,2**11}
N=[]
for i in range(5,12):
    N.append(2**i)


####### Performance de différentes méthodes de régression linéaire #########################

train_errors = np.zeros((np.size(N),4))
valid_errors =np.zeros((np.size(N),4)) 

ln_methods = [LinearRegressionMean(), LinearRegressionMedian(), LinearRegressionMajority(), LinearRegressionLeastSquares()]

j = 0
for method in ln_methods :
    i = 0
    for n in N:
        X_train=X_labeled[:n]
        y_train=y_labeled[:n]
        
        #Train by the mean constant estimator
        ln = method
        ln.fit(X_train, y_train)
        prediction_test =  ln.predict(X_test)
        prediction_train = ln.predict(X_train)
        #
        MSE_test = mean_squared_error(y_test, prediction_test)
        valid_errors[i, j] = MSE_test
        #
        MSE_train = mean_squared_error(y_train, prediction_train)
        train_errors[i, j] = MSE_train
        i = i + 1
    j = j + 1

np.savez("io/Performance_LinearRegression_Methods.npz", N, valid_errors, train_errors)


"""Exo 9"""

Learning_time = np.zeros((np.size(N), np.size(ln_methods)))

j = 0 
for method in ln_methods :
    i = 0
    for n in N:
        X_train=X_labeled[:n]
        y_train=y_labeled[:n]
        #Train by the mean constant estimator
        start_time = time.time() 
        ln = method
        ln.fit(X_train, y_train)
        pred_test =  ln.predict(X_test)
        Learning_time[i,j] = time.time() - start_time
        i = i+1
    j = j+1

np.savez("io/Matrix_Learning_time.npz", N, Learning_time)
    
    
    