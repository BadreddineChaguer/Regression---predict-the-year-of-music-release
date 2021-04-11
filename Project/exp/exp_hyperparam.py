# -*- coding: utf-8 -*-
"""
@author: CHAGUER Badreddine
"""
import numpy as np
import matplotlib.pyplot as plt
from algorithms.data_utils import load_data,randomize_data,split_data
from algorithms.linear_regression import LinearRegressionRidge, LinearRegressionMp, LinearRegressionOmp
from sklearn.metrics import mean_squared_error

X_labeled, y_labeled, X_unlabled = load_data('io/YearPredictionMSD_100.npz')
X_permute, y_permute = randomize_data(X_labeled, y_labeled)

#fixant le nombre d’exemples d’apprentissage à 500
#séparation les données étiquetées en un ensemble d’apprentissage et un ensemble de validation
X_train = X_permute[:500,:]
y_train = y_permute[:500]
X_test = X_permute[500:,:]
y_test = y_permute[500:]

#nous définissons un ensemble de valeur à tester pour les hyperparamètres.
lambda_ridge = np.arange(0, 2, 0.01)
k_max = np.arange(1,50)

#initialisation des listes des erreurs
error_valid_Ridge = []
error_valid_MP = []
error_valid_OMP = []

error_train_Ridge = []
error_train_MP = []
error_train_OMP = []


#Ridge
for i in lambda_ridge:
    ln_Ridge = LinearRegressionRidge(i) 
    ln_Ridge.fit(X_train, y_train) 
    prediction_train_Ridge = ln_Ridge.predict(X_train) 
    error_train_Ridge.append(mean_squared_error(y_train, prediction_train_Ridge))
    
    prediction_test_Ridge = ln_Ridge.predict(X_test) 
    error_valid_Ridge.append(mean_squared_error(y_test, prediction_test_Ridge)) 

plt.figure('Ridge')
plt.title("MSE according to parameters -Ridge-")
plt.semilogx(lambda_ridge, error_train_Ridge, color='red',label="Ridge training error")
plt.semilogx(lambda_ridge, error_valid_Ridge, color='green', label="Ridge validation error")
plt.legend()
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.savefig('plots/Tuning_Ridge_regression.png')
plt.close()


#Best parameter Ridge
error_train_minus_valid_Ridge = []
for i in range(len(lambda_ridge)) :
    diff = error_train_Ridge[i] - error_valid_Ridge[i]
    diff = abs(diff)
    error_train_minus_valid_Ridge.append(diff)
best_parameter_Ridge = lambda_ridge[error_train_minus_valid_Ridge.index(min(error_train_minus_valid_Ridge))]

    
print("Best parameter value of Ridge regression is :",best_parameter_Ridge)


#MP
for i in k_max:
    ln_MP = LinearRegressionMp(i) 
    ln_MP.fit(X_train, y_train) 
    prediction_train_MP = ln_MP.predict(X_train) 
    error_train_MP.append(mean_squared_error(y_train, prediction_train_MP))
    
    prediction_test_MP = ln_MP.predict(X_test) 
    error_valid_MP.append(mean_squared_error(y_test, prediction_test_MP)) 

plt.figure('MP')
plt.title("MSE according to parameters -MP-")
plt.semilogx(k_max, error_train_MP, color='red',label="MP training error")
plt.semilogx(k_max, error_valid_MP, color='green', label="MP validation error")
plt.legend()
plt.xlabel('k_max')
plt.ylabel('MSE')
plt.savefig('plots/Tuning_MP_regression.png')
plt.close()


#Best parameter MP
error_train_minus_valid_MP = []
for i in range(len(k_max)) :
    diff = error_train_MP[i] - error_valid_MP[i]
    diff = abs(diff)
    error_train_minus_valid_MP.append(diff)
best_parameter_MP = k_max[error_train_minus_valid_MP.index(min(error_train_minus_valid_MP))]

    
print("Best parameter value of MP regression is :",best_parameter_MP)



#OMP
for i in k_max:
    ln_OMP = LinearRegressionOmp(i) 
    ln_OMP.fit(X_train, y_train) 
    prediction_train_OMP = ln_OMP.predict(X_train) 
    error_train_OMP.append(mean_squared_error(y_train, prediction_train_OMP))
    
    prediction_test_OMP = ln_OMP.predict(X_test) 
    error_valid_OMP.append(mean_squared_error(y_test, prediction_test_OMP)) 

plt.figure('OMP')
plt.title("MSE according to parameters -OMP-")
plt.semilogx(k_max, error_train_OMP, color='red',label="OMP training error")
plt.semilogx(k_max, error_valid_OMP, color='green', label="OMP validation error")
plt.legend()
plt.xlabel('k_max')
plt.ylabel('MSE')
plt.savefig('plots/Tuning_OMP_regression.png')
plt.close()


#Best parameter OMP
error_train_minus_valid_OMP = []
for i in range(len(k_max)) :
    diff = error_train_OMP[i] - error_valid_OMP[i]
    diff = abs(diff)
    error_train_minus_valid_OMP.append(diff)
best_parameter_OMP = k_max[error_train_minus_valid_OMP.index(min(error_train_minus_valid_OMP))]

    
print("Best parameter value of OMP regression is :",best_parameter_OMP)




