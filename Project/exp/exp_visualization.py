import matplotlib.pyplot as plt
from algorithms.data_utils import load_data


""" Build the histogram of the years of the songs from the training set and
export the figure to the image file hist_train.png
"""

#import data
X_labeled, y_labeled, X_unlabeled= load_data('io\YearPredictionMSD_100.npz')

#visualization
plt.figure("hist")
plt.title("The years present in the train data")
plt.xlabel("Years")
plt.ylabel("number of music")
plt.hist(y_labeled)
plt.savefig('plots/hist_year.png')




                                                                    