import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist #!NB: pip install tensorflow, pip install keras -> Skriv i readme.

#load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#This i show u plot stuff, here from the training set
for i in range(100,109):  
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i])
plt.show()

# Task 1
# A:
# Design NN-based classifier using Euclidean dist. 
    # Find confusion matrix + error rate for test set
# (Should split data sets into chuncs of 1000)    
# B:
# Plot misclassified pictures
# C:
# Plot correct classified pictures.
    # For B/C: Comment in report if i agree/disagree for any of the examples







