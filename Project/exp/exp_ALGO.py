"""
@author: CHAGUER Badreddine
"""

from algorithms.linear_regression import LinearRegressionMp,LinearRegressionOmp,LinearRegressionRidge
from algorithms.ALGO import learn_all_with_Ridge, learn_all_with_Mp, learn_all_with_Omp, learn_best_predictor_and_predict_test_data
from algorithms.data_utils import load_data, split_data, randomize_data
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


'''Selection of best algorithm'''
X_labeled, y_labeled, X_unlabled = load_data('io/YearPredictionMSD_100.npz')
X_permute, y_permute = randomize_data(X_labeled, y_labeled)


train_errors = np.zeros((len(range(10)), 3))
valid1_errors = np.zeros((len(range(10)), 3))
valid2_errors = np.zeros((len(range(10)), 3))


for i in range(10):
    X_train = X_permute[:500,:]
    y_train = y_permute[:500]
    X_test = X_permute[500:,:]
    y_test = y_permute[500:]
    X_test2, y_test2, X_test1, y_test1 = split_data(X_test, y_test, 2/3)
    
    #Ridge
    ln_Ridge = learn_all_with_Ridge(X_train, y_train)
    ln_Ridge.fit(X_train, y_train)
    prediction_Ridge_train = ln_Ridge.predict(X_train)
    prediction_Ridge_test1 =  ln_Ridge.predict(X_test1)
    prediction_Ridge_test2 =  ln_Ridge.predict(X_test2)
    #rajouter les MSE trouvés dans la matrice des MSE train et test
    train_errors[i, 0] = mean_squared_error(y_train, prediction_Ridge_train)
    valid1_errors[i, 0] = mean_squared_error(y_test1, prediction_Ridge_test1)
    valid2_errors[i, 0] = mean_squared_error(y_test2, prediction_Ridge_test2)
    
    #Mp
    ln_Mp = learn_all_with_Mp(X_train, y_train)
    ln_Mp.fit(X_train, y_train)
    prediction_Mp_train = ln_Mp.predict(X_train)
    prediction_Mp_test1 =  ln_Mp.predict(X_test1)
    prediction_Mp_test2 =  ln_Mp.predict(X_test2)
    #rajouter les MSE trouvés dans la matrice des MSE train et test
    train_errors[i, 1] = mean_squared_error(y_train, prediction_Mp_train)
    valid1_errors[i, 1] = mean_squared_error(y_test1, prediction_Mp_test1)
    valid2_errors[i, 1] = mean_squared_error(y_test2, prediction_Mp_test2)
    
    #Omp
    ln_Omp = learn_all_with_Omp(X_train, y_train)
    ln_Omp.fit(X_train, y_train)
    prediction_Omp_train = ln_Omp.predict(X_train)
    prediction_Omp_test1 =  ln_Omp.predict(X_test1)
    prediction_Omp_test2 =  ln_Omp.predict(X_test2)
    #rajouter les MSE trouvés dans la matrice des MSE train et test
    train_errors[i, 2] = mean_squared_error(y_train, prediction_Omp_train)
    valid1_errors[i, 2] = mean_squared_error(y_test1, prediction_Omp_test1)
    valid2_errors[i, 2] = mean_squared_error(y_test2, prediction_Omp_test2)

# abs(train - valid2)
error_train_minus_valid2 = np.zeros((len(range(10)), 3))
for j in range(3):
    for i in range(10) :
        error_train_minus_valid2[i, j] = abs(train_errors[i, j] - valid2_errors[i, j])

# best algo
ALGOs = ["Ridge", "MP", "OMP"]
algo_best_n_times = []  #liste contient les noms des meilleurs algorithmes dans chaque itération
l = 0
for i in range(10):
    l = list(valid2_errors[i, :])
    algo_best_n_times.append(ALGOs[l.index(min(l))])
    
count_ALGO = []
count_ALGO.append(algo_best_n_times.count("Ridge"))
count_ALGO.append(algo_best_n_times.count("MP"))
count_ALGO.append(algo_best_n_times.count("OMP"))

best_ALGO = []
best_ALGO = ALGOs[count_ALGO.index(max(count_ALGO))]

print("Best algorithm after 10 iteration is :",best_ALGO)


#Plot Performance des méthodes
plt.figure("Selection of best algorithm")
plt.title("the performance of the prediction methods on the validation set 2 and learning")
plt.xlabel("iteration")
plt.ylabel("'MSE'")
plt.yscale('symlog')


plt.plot(valid2_errors[:, 0], label = "Ridge_validation2", color='red', ls =':')
plt.plot(valid2_errors[:, 1], label = "Mp_validation2 ", color='blue', ls =':')
plt.plot(valid2_errors[:, 2], label = "Omp_validation2", color='green', ls =':')

plt.plot(train_errors[:, 0], label = "Ridge_training", color='red')
plt.plot(train_errors[:, 1], label = "Mp_training", color='blue')
plt.plot(train_errors[:, 2], label = "Omp_training", color='green')

plt.legend()
plt.savefig('plots/graph_on_performance_ALGO.png')
plt.close()


"""Submit your prediction on unlabeled examples"""
learn_best_predictor_and_predict_test_data('YearPredictionMSD_100.npz')