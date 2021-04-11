# -*- coding: utf-8 -*-
"""
@author: CHAGUER Badreddine
"""

from algorithms.data_utils import load_data
import numpy as np
import matplotlib.pyplot as plt


#load data
npzfile = np.load("io\Performance_estimateur_cte_all.npz")

N = npzfile['arr_0']
validation_errors = npzfile['arr_1']
training_errors = npzfile['arr_2']

#Performance des méthodes
plt.figure("Performance of methods")
plt.title("The performance of the prediction methods on the validation set and on the learning set")
plt.xlabel("The size of the train set")
plt.ylabel("'MSE'")
plt.yscale('symlog')

plt.plot(N, validation_errors[:, 0], label = "Ln_Mean_validation ", color='red')
plt.plot(N, validation_errors[:, 1], label = "Ln_Median_validation ", color='blue')
plt.plot(N, validation_errors[:, 2], label = "Ln_Majority_validation", color='green')
plt.plot(N, validation_errors[:, 3], label = "Ln_LestSquare_validation", color='black')

plt.plot(N, training_errors[:, 0], label = "Ln_Mean_training", color='red', ls='--')
plt.plot(N, training_errors[:, 1], label = "Ln_Median_training", color='blue', ls='--')
plt.plot(N, training_errors[:, 2], label = "Ln_Majority_training", color='green', ls='--')
plt.plot(N, training_errors[:, 3], label = "Ln_LeastSquare_training", color='black', ls='--')

plt.legend()
plt.savefig('plots/graph_on_performance_methods.png')
plt.close()


"""Exo 9"""
npzfile2 = np.load('io/Matrix_Learning_time.npz')
N = npzfile2['arr_0']
Learning_time = npzfile2['arr_1']


plt.figure("The execution time of each method")
plt.title("The execution time of each method")
plt.xlabel("The size of the train set")
plt.ylabel("execution time'")

plt.plot(N, Learning_time[:, 0], label="Ln_Mean", color='red')
plt.plot(N, Learning_time[:, 1], label="Ln_Median", color='blue')
plt.plot(N, Learning_time[:, 2], label="Ln_Majority", color='green')
plt.plot(N, Learning_time[:, 3], label="Ln_LestSquare", color='black')

plt.legend()
plt.savefig('plots/Learning_time.png')
plt.close()
