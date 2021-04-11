"""
@author: CHAGUER Badreddine
"""
import numpy as np
from algorithms.data_utils import split_data, load_data, randomize_data
from sklearn.metrics import mean_squared_error
from algorithms.linear_regression import LinearRegressionRidge, LinearRegressionMp, LinearRegressionOmp


def learn_all_with_Ridge(X, y):
    """
    Ridge regression estimation with best parameter
    """
    
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples
    
    #partitionnez l’ensemble (X, y) en un ensemble d’apprentissage et un ensemble de validation 1
    X_train, y_train, X_test, y_test = split_data(X, y,2/3)
    
    #find best lambda parameter
    MSE = []
    error_train = []
    error_valid = []
    lambda_ridge = np.arange(0.3, 0.6, 0.01)
    for i in lambda_ridge:
        ln = LinearRegressionRidge(i) 
        ln.fit(X_train, y_train) 
        
        prediction_train = ln.predict(X_train) 
        error_train.append(mean_squared_error(y_train, prediction_train))
        
        prediction_test = ln.predict(X_test) 
        error_valid.append(mean_squared_error(y_test, prediction_test)) 
        
    #Best parameter Ridge
    error_train_minus_valid = []
    for i in range(len(lambda_ridge)) :
        diff = error_train[i] - error_valid[i]
        diff = abs(diff)
        error_train_minus_valid.append(diff)
    
    best_lambda = lambda_ridge[error_train_minus_valid.index(min(error_train_minus_valid))]
    
    ln_optimum = LinearRegressionRidge(best_lambda)
    ln_optimum.fit(X,y)
       
    return ln_optimum

def learn_all_with_Mp(X, y):
    """
    Matching pursuit algorithm with best parameter
    """
    
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples
    
    #partitionnez l’ensemble (X, y) en un ensemble d’apprentissage et un ensemble de validation 1
    X_train, y_train, X_test, y_test = split_data(X, y,2/3)
    
    MSE = []
    error_valid = []
    error_train = []
    k_max = np.arange(1,30)
    for k in k_max :
        ln = LinearRegressionMp(k) 
        ln.fit(X_train, y_train) 
        prediction_train = ln.predict(X_train) 
        error_train.append(mean_squared_error(y_train, prediction_train))
        
        prediction_test = ln.predict(X_test) 
        error_valid.append(mean_squared_error(y_test, prediction_test)) 
    
    #Best parameter MP
    error_train_minus_valid = []
    for i in range(len(k_max)) :
        diff = error_train[i] - error_valid[i]
        diff = abs(diff)
        error_train_minus_valid.append(diff)
        
    best_kmax = k_max[error_train_minus_valid.index(min(error_train_minus_valid))]
    
    ln_optimum = LinearRegressionMp(best_kmax)
    ln_optimum.fit(X,y)
    
    return ln_optimum


def learn_all_with_Omp(X, y):
    """
    Orthogonal matching pursuit algorithm with best parameter

    """
    
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples
    
    #partitionnez l’ensemble (X, y) en un ensemble d’apprentissage et un ensemble de validation 1
    X_train, y_train, X_test, y_test = split_data(X, y,2/3)
    
    MSE = []
    error_train = []
    error_valid = []
    k_max = np.arange(1,30)
    for k in k_max :
        ln = LinearRegressionOmp(k) 
        ln.fit(X_train, y_train) 
        
        prediction_train = ln.predict(X_train) 
        error_train.append(mean_squared_error(y_train, prediction_train))
        
        prediction_test = ln.predict(X_test) 
        error_valid.append(mean_squared_error(y_test, prediction_test)) 
        
    #Best parameter OMP
    error_train_minus_valid = []
    for i in range(len(k_max)) :
        diff = error_train[i] - error_valid[i]
        diff = abs(diff)
        error_train_minus_valid.append(diff)

    best_kmax = k_max[error_train_minus_valid.index(min(error_train_minus_valid))]
    ln_optimum = LinearRegressionOmp(best_kmax)
    ln_optimum.fit(X,y)
    
    return ln_optimum


def learn_best_predictor_and_predict_test_data(filename):
    X_labeled, y_labeled, X_unlabled = load_data(filename)
    X_permute, y_permute = randomize_data(X_labeled, y_labeled)
    X_train = X_permute[:500,:]
    y_train = y_permute[:500]
    X_test2 = X_permute[500:,:]
    y_test2 = y_permute[500:]
    
    train_errors = []
    valid2_errors = []
    
    ln_Mp = learn_all_with_Mp(X_train, y_train)
    ln_Mp.fit(X_train, y_train)
    prediction_Mp_train = ln_Mp.predict(X_train)
    prediction_Mp_test2 =  ln_Mp.predict(X_test2)
    #rajouter les MSE trouvés dans la matrice des MSE train et test
    train_errors.append(mean_squared_error(y_train, prediction_Mp_train))
    valid2_errors.append(mean_squared_error(y_test2, prediction_Mp_test2))
    print("la performance du reste des données étiquetées = ",valid2_errors)
    
    y_test = ln_Mp.predict(X_unlabled)
    np.save("io/test_prediction_results.npy", y_test)    
    
    
    
    
