# Importing the modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy


import seaborn as sns

#from keras.datasets import mnist #!NB: pip install tensorflow, pip install keras -> Skriv i readme.
#from sklearn.neighbors import KNeighborsClassifier # NB: pip install sklearn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay #,zeroOneloss
#from sklearn.cluster import KMeans

from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data

dataset_iris = np.array(data)
print(dataset_iris)

legends = iris.target_names
legendslist_iris = np.array(legends)
#print(legendslist_iris[1])


# Target vectors
t1 = np.array([1, 0, 0])
t2 = np.array([0, 1, 0])
t3 = np.array([0, 0, 1])

legends = ['Setosa', 'Versicolour', 'Virginica']
# Split data into training and testing data
#N_obsv = length(class1)
N_train = 30
N_test  = 20

Classes = 3

def load_data():
    for i in range(Classes): #3 classes
        file_name2 = './class_' + str(i+1)
        tmp = np.loadtxt(file_name2,delimiter=",")
        
    # print("\n")
    # print(tmp)
        # Extend vectors to include the ones and the class-index
        class_number = np.ones((tmp.shape[0],2)) 
        class_number[:,-1] *= i 
        tmp = np.hstack((tmp, class_number))
        print("\n")
        print(tmp)
        if i > 0:
            data = np.vstack((data, tmp))
        else:
            data = copy.deepcopy(tmp)

    return data #returns datamatrix without normalized vectors 

"""

[x1 x2 x3 x4 1 "class-index"]
[x1 x2 x3 x4 1 "class-index"]
.
.
.
[x1 x2 x3 x4 1 "class-index"]

[90, 6] - matrix

"""

def split_data(data, train_set_size):
    N = int(len(data)/Classes)

    test_set_size = N - train_set_size
    train_samples = np.zeros((Classes*train_set_size,int(len(data[0]))))
    test_samples = np.zeros((Classes*test_set_size,int(len(data[0]))))

    for i in range(Classes):
        temp = data[(N*i):((i+1)*N)]
        print(temp.shape)
        train_samples[(train_set_size*i):((i+1)*train_set_size), :] = temp[ :train_set_size,:]
        test_samples[(test_set_size*i):((i+1)*test_set_size), :] = temp[train_set_size:,:]

    return train_samples, test_samples



def train_lin_classifier(train_samples, test_samples, features, alphas, num_iterations):
    num_classes = Classes
    num_features = int(len(features))
    x = train_samples


    W = np.zeros((num_classes, num_features+1))
    t = [np.kron(np.ones(1,N_train),t1), np.kron(np.ones(1,N_train),t2), np.kron(np.ones(1,N_train),t3)];

    gradient_W_MSE = 0
    for iteration in range(num_iterations):
        g_ik = sigmoid(x, W)
        for xk in train_samples:
            """
            xk = np.array([np.transpose(x[k,:]), 1])
            zk = W*xk
            gk = sigmoid2(zk)
            tk = t[:,k]

            """
            
            #gk
            temp = np.matmul(W, (xk[ :-1]))[np.newaxis].T
            gk = sigmoid2(temp)

            #
            tk *= 0
            tk[int(xk)]


        gradient_W_MSE = gradient_W_MSE + gradient_W_MSE(gk, tk, xk)


def train_lin_classifier(train_samples, test_samples, features, tolerance):
    num_classes = 3
    num_features = len(features)
    x = train_samples

    condition = 1
    iterations = 0
    alpha = 0.001

    W = np.zeros((num_classes, num_features+1))
    t = [np.kron(np.ones(1,N_train),t1), np.kron(np.ones(1,N_train),t2), np.kron(np.ones(1,N_train),t3)];

    gradient_W_MSE = 0
    while condition: #while still over tolerance
        for k in range(num_classes*N_train): 
            xk = np.array([np.transpose(x[k,:]), 1])
            zk = W*xk
            gk = sigmoid2(zk)
            tk = t[:,k]

            gradient_W_MSE = gradient_W_MSE + gradient_W_MSE_k(gk, tk, xk)
            
        condition = np.linalg.norm(gradient_W_MSE) >= tolerance
        iterations = iterations + 1
        
        
        get_next_W(W, gradient_W_MSE, alpha)
    print(iterations)
    return W





def sigmoid(x, W): #x = samples
    exponentials = np.array([ np.exp(-(np.matmul(W, xk))) for xk in x ])
    denominators = exponentials + 1
    return (1 / denominators) #


def sigmoid2(zk):
    return np.multiply(1,1/(1+np.exp(-zk)))


def gradient_W_MSE_k(gk, tk, xk):
    #Eq (22) from compendium
    grad_gk_MSE = gk - tk
    grad_zk_MSE = np.multiply(gk,(1-gk)) #matmul?
    grad_W_zk = np.transpose(xk)
    return np.multiply(grad_gk_MSE, grad_zk_MSE)*grad_W_zk


def get_next_W(W, gradient_W_MSE, alpha):
    next_W = W + - alpha*gradient_W_MSE
    return next_W

def get_MSE(g_vec, t_vec):
    error = g_vec - t_vec
    error_T = np.transpose(error)
    return np.sum(np.matmul(error_T,error)) / 2


def main():
    print("MAAAAIN")


if __name__ == '__main__':
    main()