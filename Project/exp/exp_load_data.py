import numpy as np
from algorithms.data_utils import load_data


""" Load training data and print dimensions as well as a few coefficients
in the first and last places and at random locations.
"""

YearPredictionMSD_100 = load_data("io\YearPredictionMSD_100.npz")

print("Keys of dict data are : {}".format(list(YearPredictionMSD_100)))


for i in list(YearPredictionMSD_100):
    print("Features : {}".format(i))
    print("Type : {}".format(i.dtype))
    print("number of dimensions : {}".format(i.ndim))
    print("Shape : {}".format(i.shape))
    print("\n")
    
print("For y_labeled :")
print("first five values : {}".format(YearPredictionMSD_100[1][:5]))
print("last five values : {}".format(YearPredictionMSD_100[1][-5:]))

X_data = [0, 2]
X_data_names = ['X_labeled', 'X_unlabeled']
for i in range(2) :
    print("For feature : {}".format(X_data_names[i]))
    print("2 first coefficients of the first line : {} ".format(YearPredictionMSD_100[X_data[i]][0][:2]))
    print("2 first coefficients of the last line : {} ".format(YearPredictionMSD_100[X_data[i]][-1][:2]))
    print("last coefficient of the first line : {} ".format(YearPredictionMSD_100[X_data[i]][0][-1]))
    print("last coefficient of the last line : {} ".format(YearPredictionMSD_100[X_data[i]][-1][-1]))
    print("\n")

