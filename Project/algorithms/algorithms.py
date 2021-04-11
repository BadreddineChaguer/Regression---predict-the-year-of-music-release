#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: CHAGUER Badreddine
"""
import scipy
import numpy as np
from algorithms.data_utils import split_data
from sklearn.metrics import mean_squared_error

def normalize_dictionary(X):
    """
    Normalize matrix to have unit l2-norm columns

    Parameters
    ----------
    X : np.ndarray [n, d]
        Matrix to be normalized

    Returns
    -------
    X_normalized : np.ndarray [n, d]
        Normalized matrix
    norm_coefs : np.ndarray [d]
        Normalization coefficients (i.e., l2-norm of each column of ``X``)
    """
    # Check arguments
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    
    norm = []
    X_normed = np.zeros((np.shape(X)))
    N = range(np.shape(X)[1])
    for i in N :
         norm.append(np.linalg.norm(X[:,i], 2)) 
         X_normed[:,i]= X[:,i]/norm[i]
    
    return X_normed, norm



def ridge_regression(X, y, lambda_ridge):
    """
    Ridge regression estimation

    Minimize $\left\| X w - y\right\|_2^2 + \lambda \left\|w\right\|_2^2$
    with respect to vector $w$, for $\lambda > 0$ given a matrix $X$ and a
    vector $y$.

    Note that no constant term is added.

    Parameters
    ----------
    X : np.ndarray [n, d]
        Data matrix composed of ``n`` training examples in dimension ``d``
    y : np.ndarray [n]
        Labels of the ``n`` training examples
    lambda_ridge : float
        Non-negative penalty coefficient

    Returns
    -------
    w : np.ndarray [d]
        Estimated weight vector
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples
    

    X_hat = np.dot(np.transpose(X), X) + lambda_ridge*np.identity(np.shape(X)[1])
    return np.dot(X_hat, np.dot(np.transpose(X),y))


def mp(X, y, n_iter):
    """
    Matching pursuit algorithm

    Parameters
    ----------
    X : np.ndarray [n, d]
        Dictionary, or data matrix composed of ``n`` training examples in
        dimension ``d``. It should be normalized to have unit l2-norm
        columns before calling the algorithm.
    y : np.ndarray [n]
        Observation vector, or labels of the ``n`` training examples
    n_iter : int
        Number of iterations

    Returns
    -------
    w : np.ndarray [d]
        Estimated sparse vector
    error_norm : np.ndarray [n_iter+1]
        Vector composed of the norm of the residual error at the beginning
        of the algorithm and at the end of each iteration
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    #init du résiduel
    r= y 
    #init de w
    w = np.zeros(np.shape(X)[1])
    
    error_norm = np.zeros(n_iter + 1) 
    error_norm[0] = np.linalg.norm(r)
    
    k = 0
    for k in range(1, n_iter):
        #Calcul des corrélations du résiduel et des composantes}
        c_m = (X.T).conjugate() @ r 
        
        #Sélection de la composante la plus corrélée
        m_hat = np.argmax(np.abs(c_m))
        
        #Mise à jour de la décomposition
        w[m_hat] = w[m_hat] + c_m[m_hat] 
        
        #Mise à jour du résiduel
        r= r - c_m[m_hat] * X[ :, m_hat]
        
        error_norm[k]=np.linalg.norm(r)       
    return w, error_norm


def omp(X, y, n_iter):
    """
    Orthogonal matching pursuit algorithm

    Parameters
    ----------
    X : np.ndarray [n, d]
        Dictionary, or data matrix composed of ``n`` training examples in
        dimension ``d``. It should be normalized to have unit l2-norm
        columns before calling the algorithm.
    y : np.ndarray [n]
        Observation vector, or labels of the ``n`` training examples
    n_iter : int
        Number of iterations

    Returns
    -------
    w : np.ndarray [d]
        Estimated sparse vector
    error_norm : np.ndarray [n_iter+1]
        Vector composed of the norm of the residual error at the beginning
        of the algorithm and at the end of each iteration
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    #init du résiduel
    r= y
    
    #init de w
    w = np.zeros(np.shape(X)[1])
    
    error_norm = np.zeros(n_iter + 1) 
    error_norm[0] = np.linalg.norm(r)
    
    #support de la décomposition
    omega = []
    
    k = 0
    for k in range(1, n_iter):
        #Calcul des corrélations du résiduel et des composantes}
        c_m = (X.T).conjugate() @ r 
        
        #Sélection de la composante la plus corrélée
        m_hat = np.argmax(np.abs(c_m))
        
        #Mise à jour du support
        omega.append(m_hat)
        
        #Mise à jour de la décomposition
        w[m_hat] = np.dot(X[:,k], y)
        
        #Mise à jour du résiduel
        r = y - X[:, m_hat]* w[k]
        
        error_norm[k] = np.linalg.norm(r)       
    return w, error_norm
