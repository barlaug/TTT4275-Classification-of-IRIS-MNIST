# Importing the modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy


import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, zero_one_loss
from sklearn.datasets import load_iris


Classes = 3



def load_data():
    """Loads iris with sklearn
    
    Returns:
        Target
    """
    iris = load_iris()
    dataset_iris = np.array(iris.data)
    legendslist_iris = np.array(iris.target_names)
    target_vector = np.array(iris.target)

    return dataset_iris, legendslist_iris, target_vector


def extract_sets(data, targets, train_set_size, num_classes):
    """Splits data into training set and testing set
    Params:
       data: np.ndarray
            Full iris dataset
       train_set_size: int
            Desired size of training set
        num_classes: int
            Desired size of training set
    Returns:
        Training set, testing set: np.ndarray
    """
    N = int(len(data)/num_classes)

    test_set_size = N - train_set_size
    train_samples = np.zeros((num_classes*train_set_size,int(len(data[0]))+1))
    test_samples = np.zeros((num_classes*test_set_size,int(len(data[0]))+1))

    for i in range(num_classes):
        temp = data[(N*i):((i+1)*N)]
        # Add the ones in every x[4]
        class_number = np.ones((temp.shape[0],1))
        temp = np.hstack((temp, class_number))
        print(temp.shape)
        train_samples[(train_set_size*i):((i+1)*train_set_size), :] = temp[ :train_set_size,:]
        test_samples[(test_set_size*i):((i+1)*test_set_size), :] = temp[train_set_size:,:]
        

    return train_samples, test_samples


def sigmoid(z_k):
    """Sigmoid function. Equation (20) in compendium
    Params:
       z
    Returns:
        grad W: np.ndarray
    """
    return 1/(1+np.exp(-z_k))


def get_prediction(W, x):
    """Returns prediction matrix
    Params:
        W: np.array
            CxD+1 classifier
        x: np.array
            samples
    Returns:
        prediction_vec: np.array
            vector of predictions
    """
    z = W*x
    g = sigmoid(z)

    prediction_vec = np.zeros((len(x[0]), 1))
    for i in range(len(g[0])):
        prediction_vec[i] = np.argmax(g[:,i], axis=0)
    print('Predictions:')
    print(prediction_vec)
    return prediction_vec


def train_lin_model(train_samples, alpha, num_iterations):
    """Trains the linear model. Equations (19)-(23) in compendium implemented
    Params:
        train_samples: np.array
            The training samples
        alpha: float
            learning rate
        num_iterations: int
            Number of iterations we want to use
    Returns:
        W: np.array
            CxD+1 classifier
        MSEarray: np.array
            Array containing MSEs from the calculations. For ploting
    """
    num_classes = Classes
    num_features = int(len(train_samples[0])) #5
    print(num_features)
    print(train_samples.shape)
    classes, samples, features = train_samples.shape
    print(classes)
    print(samples)
    print(features)
    W = np.zeros((num_classes,num_features))
    t_k = np.array([[1 ,0 ,0] ,[0 ,1 ,0] ,[0 ,0 ,1]])
    g_k = np.zeros((num_classes)) # The input values we want to minimise MSE for
    g_k [0] = 1 # always start with 1
    MSEset = []
    for i in range(num_iterations):
        MSEset = 0
        grad_W_MSE = 0
        for x_k in train_samples:
            z_k = np.matmul(W, x_k.T)
            g_k = sigmoid(z_k)
            print(g_k)

            grad_g_MSE = g_k - t_k #.T?
            print(grad_g_MSE)

            grad_z_gk = np.multiply(g_k,(1 - g_k)) #element wise
            print(grad_z_gk)
            grad_W_z = x_k #.T?
            print(grad_W_z)

            grad = np.multiply(np.multiply((g_k - t_k), g_k),(1 - g_k))
            print(grad.shape)
            print(grad_W_z.shape)
            grad_W_MSE = grad_W_MSE + np.matmul(grad, grad_W_z) #np.multiply(grad_g_MSE,grad_z_gk)

            
            MSE = 0.5*np.multiply((g_k - t_k).T, (g_k - t_k))
        
        MSEset.append(MSE)
        W = W - alpha*grad_W_MSE #.T
    MSEarray = np.array(MSEset)
    return W, MSEarray


def display_CM_Error(y_pred, y_true):
    """Displays confusion matrix and classification report/errors
       for deviation between predicted and true labels
    Params:
        y_pred, y_true: np.ndarray
            Predicted and true labels/classes
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    print('Displaying confusion matrix...')
    plt.show()
    print('Classification report:')
    print(classification_report(y_true, y_pred)) # Accuracy is given in report

    #accuracy = zero_one_score(y_test, y_pred)
    #error_rate = 1 - accuracy
    #print(f"Error rate: {error_rate}%")


if __name__ == '__main__':
    num_classes = 3
    iris_data, iris_legends, iris_targets = load_data()
    #alphas = [0.5, 0.25, 0.1, 0.05, 0.005]
    alphas = [0.05]
    iterations = 5000
    train_set_size = 30
    train_set, test_set = extract_sets(iris_data, iris_targets, train_set_size, num_classes)
    for alpha in alphas:
        print("30 train, 20 test")
        W = train_lin_model(train_set[:,:], alpha, iterations) #[:,:]? 
        pred_train = get_prediction(W, train_set)
        pred_test =  get_prediction(W, test_set)
        print("Training")
        get_prediction(W, test_set)
        #display_CM_Error(y_pred, y_true)
        